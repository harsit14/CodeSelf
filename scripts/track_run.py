#!/usr/bin/env python3
"""Push an online-training artifact directory's metrics to an experiment tracker.

Backends: wandb, tensorboard, jsonl (default), null. wandb/tensorboard fall back
to jsonl when their optional dependency is missing, so this never crashes a run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.tracking import log_learning_curve_to_tracker, make_tracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Online run output dir.")
    parser.add_argument(
        "--backend",
        choices=("wandb", "tensorboard", "jsonl", "null"),
        default="jsonl",
    )
    parser.add_argument("--label", default="run")
    parser.add_argument("--project", default="codeself")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", type=Path, default=None, help="TensorBoard log dir.")
    parser.add_argument("--jsonl-path", type=Path, default=None, help="JSONL sink path.")
    args = parser.parse_args()

    jsonl_path = args.jsonl_path or (args.artifact_dir / "tracker_metrics.jsonl")
    tracker = make_tracker(
        args.backend,
        project=args.project,
        run_name=args.run_name or args.label,
        log_dir=args.log_dir or (args.artifact_dir / "tensorboard"),
        jsonl_path=jsonl_path,
    )
    cycles = log_learning_curve_to_tracker(args.artifact_dir, tracker, label=args.label)
    print(f"logged {cycles} cycles to {args.backend} tracker")
    if args.backend == "jsonl":
        print(f"metrics written to {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
