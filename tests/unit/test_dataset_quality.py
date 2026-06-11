from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import (  # noqa: E402
    Split,
    TaskRegistry,
    TaskSpec,
    TestSpec,
    check_dataset_quality,
    detect_hidden_test_leaks,
    detect_train_eval_contamination,
    normalize_text,
)


def _task(task_id: str, prompt: str, split: Split, *, hidden_code: str = "assert f(1) == 2"):
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt=prompt,
        split=split,
        entry_point="f",
        public_tests=(TestSpec(name="public", code="assert f(0) == 1"),),
        hidden_tests=(TestSpec(name="hidden", code=hidden_code),),
    )


class DatasetQualityTests(unittest.TestCase):
    def test_normalize_text_removes_punctuation_and_spacing(self) -> None:
        self.assertEqual(normalize_text("Write   add-one(x)!"), "write add one x")

    def test_detect_hidden_test_leak_in_prompt_and_public_tests(self) -> None:
        tasks = [
            TaskSpec(
                task_id="quality/leak",
                source="unit-test",
                prompt="Solve f. assert f(2) == 3",
                split=Split.TRAIN,
                entry_point="f",
                public_tests=(TestSpec(name="public", code="assert f(2) == 3"),),
                hidden_tests=(TestSpec(name="hidden", code="assert f(2) == 3"),),
            )
        ]

        findings = detect_hidden_test_leaks(tasks)

        self.assertEqual({finding.surface for finding in findings}, {"prompt", "public_tests"})
        self.assertTrue(all(finding.task_id == "quality/leak" for finding in findings))

    def test_detect_exact_and_near_train_eval_contamination(self) -> None:
        train = _task("quality/train", "Write a function that returns x plus one.", Split.TRAIN)
        exact = _task("quality/dev-exact", train.prompt, Split.DEV)
        near = _task(
            "quality/dev-near",
            "Write a Python function that returns x plus one.",
            Split.TEST_PUBLIC,
        )

        findings = detect_train_eval_contamination(
            [train, exact, near],
            near_duplicate_threshold=0.80,
        )

        self.assertEqual([finding.kind for finding in findings], ["exact_duplicate", "near_duplicate"])
        self.assertEqual(findings[0].task_id_b, "quality/dev-exact")
        self.assertEqual(findings[1].task_id_b, "quality/dev-near")

    def test_quality_report_passed_property(self) -> None:
        clean = [_task("quality/train", "Return x plus one.", Split.TRAIN)]
        dirty = [
            _task("quality/train", "Return x plus one.", Split.TRAIN),
            _task("quality/dev", "Return x plus one.", Split.DEV),
        ]

        self.assertTrue(check_dataset_quality(clean).passed)
        self.assertFalse(check_dataset_quality(dirty).passed)

    def test_validate_task_schema_reports_quality_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            TaskRegistry(
                [
                    _task("quality/train", "Return x plus one.", Split.TRAIN),
                    _task("quality/dev", "Return x plus one.", Split.DEV),
                ]
            ).to_jsonl(path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_task_schema.py"),
                    str(path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("train/eval contamination detected", completed.stdout)


if __name__ == "__main__":
    unittest.main()
