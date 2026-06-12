#!/usr/bin/env python3
"""Build a versioned task distribution from a dataset config.

The build is deterministic given the config and *fails hard* when the
contamination/leakage gate detects a problem, so a contaminated dataset can
never silently feed training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import (  # noqa: E402
    DatasetContaminationError,
    TaskRegistry,
    build_dataset,
    load_dataset_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Dataset config (JSON/YAML).")
    parser.add_argument("--output", type=Path, required=True, help="Output canonical JSONL path.")
    parser.add_argument("--manifest", type=Path, help="Optional build manifest JSON output.")
    parser.add_argument(
        "--allow-contamination",
        action="store_true",
        help="Do not exit nonzero on quality-gate findings (still reported).",
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    try:
        result = build_dataset(config)
    except DatasetContaminationError as exc:
        if not args.allow_contamination:
            print(f"ERROR: {exc}", file=sys.stderr)
            for leak in exc.report.hidden_leaks:
                print(f"  hidden-leak: {leak.task_id} [{leak.surface}] {leak.snippet}", file=sys.stderr)
            for finding in exc.report.contamination_findings:
                print(
                    f"  contamination: {finding.kind} {finding.task_id_a} <-> "
                    f"{finding.task_id_b} score={finding.score:.3f}",
                    file=sys.stderr,
                )
            return 1
        result = _build_without_gate(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    TaskRegistry(list(result.tasks)).to_jsonl(args.output)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(result.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"dataset: {config.name} version {config.version}")
    print(f"wrote {len(result.tasks)} tasks to {args.output}")
    print(f"fingerprint: {result.fingerprint}")
    for split_name, count in result.manifest()["splits"].items():
        print(f"  {split_name}: {count}")
    print(f"auto-split tasks: {len(result.auto_split_task_ids)}")
    if result.tasks_without_hidden:
        print(f"WARNING: {len(result.tasks_without_hidden)} task(s) have no hidden tests")
    print(f"quality gate passed: {result.quality_report.passed}")
    return 0


def _build_without_gate(config):
    from dataclasses import replace

    return build_dataset(replace(config, enforce_quality_gate=False))


if __name__ == "__main__":
    raise SystemExit(main())
