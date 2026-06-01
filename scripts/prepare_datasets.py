#!/usr/bin/env python3
"""Convert local benchmark files into canonical CodeSelf task JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import (  # noqa: E402
    Split,
    SplitFractions,
    TaskRegistry,
    apply_test_sidecar,
    assign_splits,
    dataset_fingerprint,
    load_humaneval_like,
    load_mbpp,
    load_task_specs,
    split_counts,
    write_split_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("mbpp", "humaneval", "canonical"), required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input JSON/JSONL benchmark file.")
    parser.add_argument("--output", type=Path, required=True, help="Output canonical JSONL path.")
    parser.add_argument("--source", default=None, help="Source label stored on tasks.")
    parser.add_argument("--split", choices=[split.value for split in Split], default=Split.TRAIN.value)
    parser.add_argument("--sidecar", type=Path, help="Optional sidecar JSON/JSONL with extra tests.")
    parser.add_argument("--assign-splits", action="store_true", help="Assign deterministic splits.")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--dev-frac", type=float, default=0.1)
    parser.add_argument("--test-public-frac", type=float, default=0.1)
    parser.add_argument("--test-private-frac", type=float, default=0.0)
    parser.add_argument("--manifest", type=Path, help="Optional split manifest JSON output.")
    args = parser.parse_args()

    source = args.source or args.format
    if args.format == "mbpp":
        tasks = load_mbpp(args.input, split=args.split, source=source)
    elif args.format == "humaneval":
        tasks = load_humaneval_like(args.input, split=args.split, source=source)
    else:
        tasks = load_task_specs(args.input)

    if args.sidecar:
        tasks = apply_test_sidecar(tasks, args.sidecar)

    if args.assign_splits:
        tasks = assign_splits(
            tasks,
            fractions=SplitFractions(
                train=args.train_frac,
                dev=args.dev_frac,
                test_public=args.test_public_frac,
                test_private=args.test_private_frac,
            ),
            seed=args.seed,
        )

    registry = TaskRegistry(tasks)
    registry.to_jsonl(args.output)

    if args.manifest:
        write_split_manifest(
            tasks,
            args.manifest,
            dataset_name=source,
            version="local",
            seed=args.seed,
        )

    print(f"wrote {len(tasks)} tasks to {args.output}")
    print(f"fingerprint: {dataset_fingerprint(tasks)}")
    for split_name, count in split_counts(tasks).items():
        print(f"{split_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
