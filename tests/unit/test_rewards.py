from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import ResourceLimits, Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402
from codeself.execution import SandboxedTestRunner  # noqa: E402
from codeself.rewards import CompositeRewardScorer, measure_quality, score_correctness  # noqa: E402


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="reward/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.DEV,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
        resource_limits=ResourceLimits(timeout_seconds=1.0, memory_mb=256),
    )


class RewardTests(unittest.TestCase):
    def test_passing_solution_gets_full_available_correctness_reward(self) -> None:
        result = SandboxedTestRunner().run(_task(), "def add_one(x):\n    return x + 1")
        breakdown = score_correctness(result)

        self.assertEqual(breakdown.reward, 1.0)
        self.assertEqual(breakdown.total_penalty, 0.0)
        self.assertEqual(len(breakdown.components), 4)
        self.assertFalse(
            next(component for component in breakdown.components if component.name == "robustness_test_fraction").available
        )

    def test_public_failure_keeps_partial_syntax_import_credit(self) -> None:
        result = SandboxedTestRunner().run(_task(), "def add_one(x):\n    return x")
        breakdown = score_correctness(result)

        self.assertGreater(breakdown.reward, 0.0)
        self.assertLess(breakdown.reward, 1.0)
        self.assertEqual(breakdown.total_penalty, 0.0)

    def test_timeout_applies_penalty(self) -> None:
        task = TaskSpec(
            task_id="reward/spin",
            source="unit-test",
            prompt="Write spin().",
            split=Split.DEV,
            entry_point="spin",
            public_tests=(TestSpec(name="public", code="assert spin() == 1"),),
            resource_limits=ResourceLimits(timeout_seconds=0.2, memory_mb=256),
        )
        result = SandboxedTestRunner().run(task, "def spin():\n    while True:\n        pass")
        breakdown = score_correctness(result)

        self.assertLess(breakdown.reward, 0.0)
        self.assertIn("timeout", [penalty.name for penalty in breakdown.penalties])

    def test_composite_reward_attaches_quality_and_efficiency_metrics(self) -> None:
        code = "def add_one(x):\n    return x + 1"
        result = SandboxedTestRunner().run(_task(), code)
        breakdown = CompositeRewardScorer().score(
            result,
            solution_code=code,
            reference_duration_seconds=result.duration_seconds * 2,
        )

        self.assertIn("quality_score", breakdown.metrics)
        self.assertIn("efficiency_duration_seconds", breakdown.metrics)
        self.assertIn("efficiency_relative_runtime_score", breakdown.metrics)

    def test_quality_metric_rejects_invalid_syntax(self) -> None:
        metrics = measure_quality("def bad(:")

        self.assertFalse(metrics.syntax_valid)
        self.assertEqual(metrics.quality_score, 0.0)

    def test_audit_reward_script_outputs_json_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            solution_path = Path(tmpdir) / "solution.py"
            TaskRegistry([_task()]).to_jsonl(task_path)
            solution_path.write_text("def add_one(x):\n    return x + 1\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_reward.py"),
                    "--tasks",
                    str(task_path),
                    "--solution-file",
                    str(solution_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["reward_name"], "reward_v0_correctness")
        self.assertEqual(payload["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
