from __future__ import annotations

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
    bootstrap_delta_ci,
    compare_rollouts,
    exact_mcnemar,
    paired_task_outcomes,
)


def _task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TEST_PUBLIC,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


def _rollouts(task_ids: list[str], passing: set[str]):
    records = []
    for task_id in task_ids:
        completion = (
            "```python\ndef add_one(x):\n    return x + 1\n```"
            if task_id in passing
            else "```python\ndef add_one(x):\n    return x\n```"
        )
        records.extend(
            generate_rollouts(
                [_task(task_id)],
                generator=StaticGenerator(completion),
                prompt_template=DIRECT_SOLUTION_TEMPLATE,
                samples_per_task=1,
                seed=1,
                max_new_tokens=128,
                temperature=0.0,
                top_p=1.0,
                include_hidden=True,
            )
        )
    return records


class StatisticalTests(unittest.TestCase):
    def test_paired_outcomes_and_mcnemar_counts(self) -> None:
        task_ids = ["task/a", "task/b", "task/c", "task/d"]
        base = _rollouts(task_ids, passing={"task/a", "task/b"})
        candidate = _rollouts(task_ids, passing={"task/b", "task/c"})

        outcomes = paired_task_outcomes(base, candidate)
        result = exact_mcnemar(outcomes)

        self.assertEqual(result.both_passed, 1)
        self.assertEqual(result.both_failed, 1)
        self.assertEqual(result.base_only, 1)
        self.assertEqual(result.candidate_only, 1)
        self.assertEqual(result.discordant, 2)
        self.assertEqual(result.p_value, 1.0)

    def test_compare_rollouts_reports_delta_and_bootstrap_ci(self) -> None:
        task_ids = ["task/a", "task/b", "task/c", "task/d"]
        base = _rollouts(task_ids, passing={"task/a"})
        candidate = _rollouts(task_ids, passing={"task/a", "task/b", "task/c"})

        summary = compare_rollouts(base, candidate, bootstrap_samples=200, seed=1)

        self.assertEqual(summary.task_count, 4)
        self.assertAlmostEqual(summary.base_pass_rate, 0.25)
        self.assertAlmostEqual(summary.candidate_pass_rate, 0.75)
        self.assertAlmostEqual(summary.delta_pp, 50.0)
        self.assertGreaterEqual(summary.bootstrap_delta.upper, summary.bootstrap_delta.lower)

    def test_bootstrap_requires_outcomes(self) -> None:
        with self.assertRaises(ValueError):
            bootstrap_delta_ci([])

    def test_compare_rollouts_script_writes_report(self) -> None:
        task_ids = ["task/a", "task/b", "task/c"]
        base = _rollouts(task_ids, passing={"task/a"})
        candidate = _rollouts(task_ids, passing={"task/a", "task/b"})
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir) / "base.jsonl"
            candidate_path = Path(tmpdir) / "candidate.jsonl"
            output_path = Path(tmpdir) / "comparison.md"
            write_rollouts_jsonl(base, base_path)
            write_rollouts_jsonl(candidate, candidate_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_rollouts.py"),
                    "--base-rollouts",
                    str(base_path),
                    "--candidate-rollouts",
                    str(candidate_path),
                    "--output",
                    str(output_path),
                    "--bootstrap-samples",
                    "100",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            report = output_path.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Final Paired Evaluation Report", report)
        self.assertIn("mcnemar_p_value", completed.stdout)


if __name__ == "__main__":
    unittest.main()
