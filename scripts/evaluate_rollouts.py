#!/usr/bin/env python3
"""Evaluate rollout JSONL files and write baseline reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import read_rollouts_jsonl  # noqa: E402
from codeself.evaluation import (  # noqa: E402
    approximate_minimum_detectable_effect,
    evaluate_rollouts,
    write_evaluation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True, help="Input rollout JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Output .md or .json report.")
    parser.add_argument("--ks", default="1,5,10", help="Comma-separated pass@k values.")
    parser.add_argument("--power-output", type=Path, help="Optional power-analysis JSON output.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    ks = tuple(int(value) for value in args.ks.split(",") if value.strip())
    records = read_rollouts_jsonl(args.rollouts)
    summary = evaluate_rollouts(records, ks=ks)
    write_evaluation_report(summary, args.output)

    print(f"wrote evaluation report to {args.output}")
    print(f"rollouts: {summary.rollout_count}")
    print(f"tasks: {summary.task_count}")
    print(f"execution_pass_rate: {summary.execution_pass_rate:.4f}")
    print(f"parse_failure_rate: {summary.parse_failure_rate:.4f}")
    print(f"informative_prompt_fraction: {summary.informative_prompt_fraction:.4f}")
    for key, value in sorted(summary.pass_at.items()):
        print(f"{key}: {value:.4f}")

    if args.power_output:
        baseline = summary.pass_at.get("pass@1", summary.execution_pass_rate)
        power = approximate_minimum_detectable_effect(
            task_count=max(1, summary.task_count),
            baseline_pass_rate=baseline,
            alpha=args.alpha,
        )
        args.power_output.parent.mkdir(parents=True, exist_ok=True)
        args.power_output.write_text(
            json.dumps(power.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote power analysis to {args.power_output}")
        print(f"minimum_detectable_effect_pp: {power.minimum_detectable_effect_pp:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
