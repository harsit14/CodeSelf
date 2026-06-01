"""Small training logging helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlMetricWriter:
    """Append metric dictionaries to a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, metrics: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True))
            handle.write("\n")


def write_checkpoint_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a checkpoint manifest JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
