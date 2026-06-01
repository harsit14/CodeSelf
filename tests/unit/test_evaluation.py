from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import DIRECT_SOLUTION_TEMPLATE, StaticGenerator, generate_rollouts, write_rollouts_jsonl  # noqa: E402
from codeself.datasets import Split, TaskSpec, TestSpec  # noqa: E402
from codeself.evaluation import (  # noqa: E402
    approximate_minimum_detectable_effect,
    estimate_pass_at_k,
    evaluate_rollouts,
)


def _task(task_id: str = "eval/add-one") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TEST_PUBLIC,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


def _records_for_completion(task: TaskSpec, completion: str, samples: int = 1):
    return generate_rollouts(
        [task],
        generator=StaticGenerator(completion),
        prompt_template=DIRECT_SOLUTION_TEMPLATE,
        samples_per_task=samples,
        seed=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        include_hidden=True,
    )


class EvaluationTests(unittest.TestCase):
    def test_pass_at_k_estimator(self) -> None:
        self.assertAlmostEqual(estimate_pass_at_k(10, 0, 1), 0.0)
        self.assertAlmostEqual(estimate_pass_at_k(10, 10, 1), 1.0)
        self.assertAlmostEqual(estimate_pass_at_k(10, 1, 1), 0.1)
        self.assertAlmostEqual(estimate_pass_at_k(10, 1, 5), 0.5)

    def test_evaluate_rollouts_summarizes_pass_and_informative_fraction(self) -> None:
        passing = _records_for_completion(
            _task("eval/pass"),
            "```python\ndef add_one(x):\n    return x + 1\n```",
            samples=2,
        )
        failing = _records_for_completion(
            _task("eval/fail"),
            "```python\ndef add_one(x):\n    return x\n```",
            samples=2,
        )
        mixed = passing[:1] + failing[:1]
        object.__setattr__(mixed[0], "task_id", "eval/mixed")
        object.__setattr__(mixed[1], "task_id", "eval/mixed")

        summary = evaluate_rollouts(passing + failing + mixed, ks=(1, 2))

        self.assertEqual(summary.task_count, 3)
        self.assertAlmostEqual(summary.execution_pass_rate, 0.5)
        self.assertAlmostEqual(summary.informative_prompt_fraction, 1 / 3)
        self.assertIn("pass@1", summary.pass_at)
        self.assertIn("pass@2", summary.pass_at)

    def test_power_analysis_returns_percentage_point_effect(self) -> None:
        result = approximate_minimum_detectable_effect(task_count=100, baseline_pass_rate=0.5)

        self.assertGreater(result.minimum_detectable_effect_pp, 0)
        self.assertLess(result.minimum_detectable_effect_pp, 30)

    def test_evaluate_rollouts_script_writes_markdown_and_power_json(self) -> None:
        records = _records_for_completion(
            _task(),
            "```python\ndef add_one(x):\n    return x + 1\n```",
            samples=3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            rollout_path = Path(tmpdir) / "rollouts.jsonl"
            report_path = Path(tmpdir) / "report.md"
            power_path = Path(tmpdir) / "power.json"
            write_rollouts_jsonl(records, rollout_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_rollouts.py"),
                    "--rollouts",
                    str(rollout_path),
                    "--output",
                    str(report_path),
                    "--power-output",
                    str(power_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Baseline Evaluation Report", report_path.read_text(encoding="utf-8"))
            power_payload = json.loads(power_path.read_text(encoding="utf-8"))

        self.assertIn("minimum_detectable_effect_pp", power_payload)
        self.assertIn("pass@1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
