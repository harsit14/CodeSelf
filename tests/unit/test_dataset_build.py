from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.config import load_config_file  # noqa: E402
from codeself.datasets import (  # noqa: E402
    DatasetConfig,
    DatasetContaminationError,
    DatasetSourceConfig,
    Split,
    TaskRegistry,
    TaskSpec,
    TestSpec,
    auto_split_visible_hidden,
    build_dataset,
    load_dataset_config,
)
from codeself.datasets.schemas import TestVisibility  # noqa: E402


def _task(task_id: str, *, public: list[str], hidden: list[str]) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt=f"Solve {task_id}.",
        split=Split.TRAIN,
        entry_point="f",
        public_tests=tuple(
            TestSpec(name=f"public-{i}", code=c, visibility=TestVisibility.PUBLIC)
            for i, c in enumerate(public, 1)
        ),
        hidden_tests=tuple(
            TestSpec(name=f"hidden-{i}", code=c, visibility=TestVisibility.HIDDEN)
            for i, c in enumerate(hidden, 1)
        ),
    )


class AutoSplitTests(unittest.TestCase):
    def test_holds_out_public_tests_as_hidden_when_none_exist(self) -> None:
        task = _task("auto/split", public=["assert f(1)", "assert f(2)", "assert f(3)"], hidden=[])
        new_task, was_split = auto_split_visible_hidden(task, visible_tests_per_task=1)

        self.assertTrue(was_split)
        self.assertEqual(len(new_task.public_tests), 1)
        self.assertEqual(len(new_task.hidden_tests), 2)
        self.assertTrue(all(t.visibility == TestVisibility.HIDDEN for t in new_task.hidden_tests))

    def test_keeps_existing_hidden_tests_unchanged(self) -> None:
        task = _task("auto/keep", public=["assert f(1)"], hidden=["assert f(2)"])
        new_task, was_split = auto_split_visible_hidden(task)

        self.assertFalse(was_split)
        self.assertEqual(new_task.hidden_tests, task.hidden_tests)

    def test_does_not_split_when_too_few_tests(self) -> None:
        task = _task("auto/single", public=["assert f(1)"], hidden=[])
        new_task, was_split = auto_split_visible_hidden(task, visible_tests_per_task=1)

        self.assertFalse(was_split)
        self.assertEqual(len(new_task.public_tests), 1)
        self.assertEqual(len(new_task.hidden_tests), 0)


class BuildDatasetGateTests(unittest.TestCase):
    def _config(self, path: Path, **overrides) -> DatasetConfig:
        kwargs = dict(
            name="unit",
            version="v0",
            sources=(DatasetSourceConfig(format="canonical", path=str(path)),),
            assign_splits=False,
        )
        kwargs.update(overrides)
        return DatasetConfig(**kwargs)

    def test_build_raises_on_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            train = _task("dup/train", public=["assert f(1)"], hidden=["assert f(2)"])
            from dataclasses import replace

            dev = replace(train, task_id="dup/dev", split=Split.DEV)
            TaskRegistry([train, dev]).to_jsonl(path)

            with self.assertRaises(DatasetContaminationError):
                build_dataset(self._config(path))

    def test_build_can_skip_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            from dataclasses import replace

            train = _task("dup/train", public=["assert f(1)"], hidden=["assert f(2)"])
            dev = replace(train, task_id="dup/dev", split=Split.DEV)
            TaskRegistry([train, dev]).to_jsonl(path)

            result = build_dataset(self._config(path, enforce_quality_gate=False))
            self.assertFalse(result.quality_report.passed)
            self.assertEqual(len(result.tasks), 2)

    def test_build_clean_dataset_passes_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            tasks = [
                _task("clean/a", public=["assert f(1)"], hidden=["assert f(2)"]),
                _task("clean/b", public=["assert g(1)"], hidden=["assert g(2)"]),
            ]
            TaskRegistry(tasks).to_jsonl(path)

            result = build_dataset(self._config(path))
            self.assertTrue(result.quality_report.passed)
            self.assertEqual(len(result.tasks), 2)
            manifest = result.manifest()
            self.assertEqual(manifest["dataset"]["task_count"], 2)
            self.assertEqual(manifest["dataset"]["name"], "unit")

    def test_duplicate_task_id_across_sources_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            tasks = [_task("same/id", public=["assert f(1)"], hidden=["assert f(2)"])]
            TaskRegistry(tasks).to_jsonl(path)
            config = DatasetConfig(
                name="dup",
                version="v0",
                sources=(
                    DatasetSourceConfig(format="canonical", path=str(path)),
                    DatasetSourceConfig(format="canonical", path=str(path)),
                ),
                assign_splits=False,
            )
            with self.assertRaises(Exception):
                build_dataset(config)


class DebugDatasetConfigTests(unittest.TestCase):
    def test_debug_config_builds_clean_distribution(self) -> None:
        config = load_dataset_config(ROOT / "configs" / "datasets" / "debug.dataset.yaml")
        result = build_dataset(config)

        self.assertEqual(config.name, "codeself-debug")
        self.assertEqual(len(result.tasks), 12)
        self.assertTrue(result.quality_report.passed)
        splits = result.manifest()["splits"]
        self.assertEqual(sum(splits.values()), 12)
        self.assertGreater(splits.get("train", 0), 0)
        for task in result.tasks:
            self.assertTrue(task.hidden_tests, f"{task.task_id} should have hidden tests")


class SimpleYamlMultiKeyListTests(unittest.TestCase):
    def test_parses_multi_key_block_mapping_list_items(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "c.yaml"
            path.write_text(
                "dataset:\n"
                "  name: x\n"
                "  sources:\n"
                "    - format: canonical\n"
                "      path: a.jsonl\n"
                "      public_tests_per_task: 2\n"
                "    - format: mbpp\n"
                "      path: b.jsonl\n",
                encoding="utf-8",
            )
            loaded = load_config_file(path)

        sources = loaded["dataset"]["sources"]
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["format"], "canonical")
        self.assertEqual(sources[0]["path"], "a.jsonl")
        self.assertEqual(sources[0]["public_tests_per_task"], 2)
        self.assertEqual(sources[1]["format"], "mbpp")

    def test_scalar_list_items_still_parse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "c.yaml"
            path.write_text("items:\n  - one\n  - two\n", encoding="utf-8")
            loaded = load_config_file(path)
        self.assertEqual(loaded["items"], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
