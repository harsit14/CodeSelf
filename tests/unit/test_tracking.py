from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.tracking import (  # noqa: E402
    JsonlTracker,
    NullTracker,
    make_tracker,
)


class TrackerFactoryTests(unittest.TestCase):
    def test_null_backend(self) -> None:
        tracker = make_tracker("null")
        self.assertIsInstance(tracker, NullTracker)
        tracker.log({"x": 1.0}, step=0)  # no-op
        tracker.finish()

    def test_jsonl_backend_writes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "m.jsonl"
            tracker = make_tracker("jsonl", jsonl_path=path)
            self.assertIsInstance(tracker, JsonlTracker)
            tracker.log_config({"lr": 0.001})
            tracker.log({"reward": 0.5, "kl": 0.01}, step=1)
            tracker.log({"reward": 0.7}, step=2)
            tracker.finish()

            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(lines[0]["config"]["lr"], 0.001)
        self.assertEqual(lines[1]["step"], 1)
        self.assertEqual(lines[1]["reward"], 0.5)
        self.assertEqual(lines[2]["step"], 2)

    def test_wandb_backend_falls_back_to_jsonl_when_missing(self) -> None:
        # wandb is not installed in CI; make_tracker must degrade gracefully.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "m.jsonl"
            tracker = make_tracker("wandb", jsonl_path=path)
            tracker.log({"reward": 1.0}, step=0)
            tracker.finish()
            self.assertTrue(path.exists())

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_tracker("mlflow")

    def test_to_scalar_handles_bool_and_numbers(self) -> None:
        from codeself.tracking.trackers import _to_scalar

        self.assertEqual(_to_scalar(True), 1.0)
        self.assertEqual(_to_scalar(3), 3.0)
        self.assertEqual(_to_scalar(2.5), 2.5)


if __name__ == "__main__":
    unittest.main()
