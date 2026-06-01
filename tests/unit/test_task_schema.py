from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import ResourceLimits, Split, TaskRegistry, TaskSpec, TestSpec


class TaskSchemaTests(unittest.TestCase):
    def test_task_round_trip(self) -> None:
        task = TaskSpec(
            task_id="toy/add-one",
            source="unit-test",
            prompt="Write add_one(x).",
            split=Split.TRAIN,
            entry_point="add_one",
            public_tests=(TestSpec(name="public-basic", code="assert add_one(1) == 2"),),
            hidden_tests=(TestSpec(name="hidden-zero", code="assert add_one(0) == 1"),),
            tags=("arithmetic", "toy"),
            resource_limits=ResourceLimits(timeout_seconds=1.5, memory_mb=128),
            metadata={"license": "test-only"},
        )

        restored = TaskSpec.from_dict(task.to_dict())

        self.assertEqual(restored.task_id, task.task_id)
        self.assertEqual(restored.split, Split.TRAIN)
        self.assertEqual(len(restored.all_tests), 2)
        self.assertEqual(restored.public_tests[0].visibility.value, "public")
        self.assertEqual(restored.hidden_tests[0].visibility.value, "hidden")
        self.assertEqual(restored.resource_limits.memory_mb, 128)

    def test_registry_rejects_duplicate_task_ids(self) -> None:
        task = TaskSpec(
            task_id="duplicate",
            source="unit-test",
            prompt="Return 1.",
            split=Split.DEV,
        )
        registry = TaskRegistry([task])

        with self.assertRaises(ValueError):
            registry.add(task)

    def test_registry_jsonl_round_trip(self) -> None:
        task = TaskSpec(
            task_id="jsonl/one",
            source="unit-test",
            prompt="Return x.",
            split=Split.TEST_PRIVATE,
        )
        registry = TaskRegistry([task])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            registry.to_jsonl(path)
            restored = TaskRegistry.from_jsonl(path)

        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.get("jsonl/one").split, Split.TEST_PRIVATE)

    def test_invalid_resource_limits_fail_fast(self) -> None:
        with self.assertRaises(ValueError):
            ResourceLimits(timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
