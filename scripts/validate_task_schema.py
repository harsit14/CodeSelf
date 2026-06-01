#!/usr/bin/env python3
"""Validate a task JSONL file against the CodeSelf task schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import TaskRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a task JSONL file.")
    args = parser.parse_args()

    registry = TaskRegistry.from_jsonl(args.path)
    split_counts: dict[str, int] = {}
    hidden_leaks: list[str] = []
    for task in registry:
        split_counts[task.split.value] = split_counts.get(task.split.value, 0) + 1
        prompt_surface = f"{task.prompt}\n{task.starter_code}"
        for test in task.hidden_tests:
            if test.code.strip() and test.code.strip() in prompt_surface:
                hidden_leaks.append(f"{task.task_id}:{test.name}")

    print(f"validated {len(registry)} tasks from {args.path}")
    for split, count in sorted(split_counts.items()):
        print(f"{split}: {count}")
    if hidden_leaks:
        print("hidden test leakage detected:")
        for leak in hidden_leaks:
            print(f"- {leak}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
