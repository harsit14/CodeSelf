from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import (  # noqa: E402
    Split,
    SplitFractions,
    apply_test_sidecar,
    assign_splits,
    dataset_fingerprint,
    load_humaneval_like,
    load_mbpp,
    split_counts,
)


class DatasetLoaderTests(unittest.TestCase):
    def test_mbpp_loader_converts_tests_and_entry_point(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbpp.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": 11,
                            "text": "Write a function add_one(x).",
                            "test_list": [
                                "assert add_one(1) == 2",
                                "assert add_one(0) == 1",
                            ],
                            "challenge_test_list": ["assert add_one(-1) == 0"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            tasks = load_mbpp(path, split=Split.TRAIN, source="mbpp")

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, "mbpp/11")
        self.assertEqual(tasks[0].entry_point, "add_one")
        self.assertEqual(len(tasks[0].public_tests), 1)
        self.assertEqual(len(tasks[0].hidden_tests), 2)

    def test_humaneval_like_loader_reads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "humaneval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "task_id": "HumanEval/0",
                        "prompt": "def add_one(x):\n    pass",
                        "entry_point": "add_one",
                        "test": "def check(candidate):\n    assert candidate(1) == 2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            tasks = load_humaneval_like(path, split=Split.TEST_PUBLIC, source="humaneval")

        self.assertEqual(tasks[0].task_id, "HumanEval/0")
        self.assertEqual(tasks[0].split, Split.TEST_PUBLIC)
        self.assertEqual(len(tasks[0].hidden_tests), 1)
        self.assertIn("check(add_one)", tasks[0].hidden_tests[0].code)

    def test_hidden_test_sidecar_can_override_by_unscoped_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "mbpp.json"
            sidecar_path = Path(tmpdir) / "sidecar.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": 7,
                            "text": "Return x.",
                            "test_list": ["assert identity(1) == 1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            sidecar_path.write_text(
                json.dumps({"7": {"hidden_tests": ["assert identity('a') == 'a'"]}}),
                encoding="utf-8",
            )

            tasks = apply_test_sidecar(load_mbpp(dataset_path), sidecar_path)

        self.assertEqual(tasks[0].task_id, "mbpp/7")
        self.assertEqual(len(tasks[0].hidden_tests), 1)
        self.assertIn("'a'", tasks[0].hidden_tests[0].code)

    def test_assign_splits_is_deterministic(self) -> None:
        tasks = []
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mbpp.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": index,
                            "text": f"Task {index}",
                            "test_list": [f"assert f_{index}() == {index}"],
                        }
                        for index in range(10)
                    ]
                ),
                encoding="utf-8",
            )
            tasks = load_mbpp(path)

        first = assign_splits(
            tasks,
            fractions=SplitFractions(train=0.6, dev=0.2, test_public=0.2),
            seed=123,
        )
        second = assign_splits(
            tasks,
            fractions=SplitFractions(train=0.6, dev=0.2, test_public=0.2),
            seed=123,
        )

        self.assertEqual([task.to_dict() for task in first], [task.to_dict() for task in second])
        self.assertEqual(split_counts(first), {"dev": 2, "test_public": 2, "train": 6})
        self.assertEqual(dataset_fingerprint(first), dataset_fingerprint(second))

    def test_prepare_datasets_script_writes_canonical_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "mbpp.json"
            output_path = Path(tmpdir) / "out.jsonl"
            manifest_path = Path(tmpdir) / "manifest.json"
            input_path.write_text(
                json.dumps(
                    [
                        {
                            "task_id": 1,
                            "text": "Return one.",
                            "test_list": ["assert one() == 1"],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "prepare_datasets.py"),
                    "--format",
                    "mbpp",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--manifest",
                    str(manifest_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertIn("wrote 1 tasks", completed.stdout)


if __name__ == "__main__":
    unittest.main()
