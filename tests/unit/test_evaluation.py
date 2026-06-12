from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    DIRECT_SOLUTION_TEMPLATE,
    ParsedCompletion,
    ParseStatus,
    StaticGenerator,
    generate_rollouts,
    write_rollouts_jsonl,
)
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

    def test_evaluate_rollouts_groups_and_caps_sample_budgets(self) -> None:
        easy = _records_for_completion(
            _task("eval/easy"),
            "```python\ndef add_one(x):\n    return x + 1\n```",
            samples=3,
        )
        hard = _records_for_completion(
            _task("eval/hard"),
            "```python\ndef add_one(x):\n    return x\n```",
            samples=3,
        )
        for index, record in enumerate(easy):
            record.metadata["difficulty"] = "easy"
            record.metadata["response_tokens"] = index + 2
        for index, record in enumerate(hard):
            record.metadata["difficulty"] = "hard"
            record.metadata["response_tokens"] = index + 5
        object.__setattr__(hard[0], "raw_completion", "repeat " * 24)
        object.__setattr__(
            hard[0],
            "parsed",
            ParsedCompletion(
                raw_text="repeat " * 24,
                code="x\nx\nx\nx\n",
                status=ParseStatus.OK,
                parser="unit",
            ),
        )

        summary = evaluate_rollouts(
            easy + hard,
            ks=(1, 2),
            max_samples_per_task=2,
            group_by=("metadata.difficulty",),
        )
        groups = {group.group_value: group for group in summary.group_summaries}
        tasks = {task.task_id: task for task in summary.task_summaries}

        self.assertEqual(summary.rollout_count, 4)
        self.assertEqual(summary.response_token_max, 6)
        self.assertAlmostEqual(summary.degenerate_output_rate, 0.25)
        self.assertEqual(set(groups), {"easy", "hard"})
        self.assertEqual(groups["easy"].rollout_count, 2)
        self.assertAlmostEqual(groups["easy"].pass_at["pass@1"], 1.0)
        self.assertAlmostEqual(groups["hard"].pass_at["pass@1"], 0.0)
        self.assertAlmostEqual(groups["hard"].degenerate_output_rate, 0.5)
        self.assertEqual(tasks["eval/easy"].total_samples, 2)
        self.assertAlmostEqual(tasks["eval/easy"].pass_at["pass@2"], 1.0)
        self.assertIn("Group Summary", summary.to_markdown())

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
                    "--group-by",
                    "metadata.seed",
                    "--max-samples-per-task",
                    "2",
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
        self.assertIn("degenerate_output_rate", completed.stdout)
        self.assertIn("group[metadata.seed=1]", completed.stdout)


if __name__ == "__main__":
    unittest.main()
