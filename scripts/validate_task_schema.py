#!/usr/bin/env python3
"""Validate a task JSONL file against the CodeSelf task schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import TaskRegistry, check_dataset_quality  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to a task JSONL file.")
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=0.92,
        help="Train/eval near-duplicate threshold over normalized task text.",
    )
    args = parser.parse_args()

    registry = TaskRegistry.from_jsonl(args.path)
    split_counts: dict[str, int] = {}
    tasks = list(registry)
    for task in tasks:
        split_counts[task.split.value] = split_counts.get(task.split.value, 0) + 1
    quality = check_dataset_quality(
        tasks,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )

    print(f"validated {len(registry)} tasks from {args.path}")
    for split, count in sorted(split_counts.items()):
        print(f"{split}: {count}")
    if quality.hidden_leaks:
        print("hidden test leakage detected:")
        for leak in quality.hidden_leaks:
            print(f"- {leak.task_id}:{leak.test_name} in {leak.surface}")
    if quality.contamination_findings:
        print("train/eval contamination detected:")
        for finding in quality.contamination_findings:
            print(
                f"- {finding.kind}: {finding.task_id_a} ({finding.split_a}) <> "
                f"{finding.task_id_b} ({finding.split_b}), score={finding.score:.4f}"
            )
    if not quality.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
