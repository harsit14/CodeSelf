#!/usr/bin/env python3
"""Compare base and candidate rollout files with paired statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.evaluation import compare_rollout_files, write_comparison_report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rollouts", type=Path, required=True)
    parser.add_argument("--candidate-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output .md or .json report.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    summary = compare_rollout_files(
        args.base_rollouts,
        args.candidate_rollouts,
        sample_index=args.sample_index,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
        confidence=args.confidence,
    )
    write_comparison_report(summary, args.output)
    print(f"wrote paired comparison report to {args.output}")
    print(f"tasks: {summary.task_count}")
    print(f"base_pass@1: {summary.base_pass_rate:.4f}")
    print(f"candidate_pass@1: {summary.candidate_pass_rate:.4f}")
    print(f"delta_pp: {summary.delta_pp:.2f}")
    print(f"mcnemar_p_value: {summary.mcnemar.p_value:.6f}")
    print(f"permutation_p_value: {summary.permutation.p_value:.6f}")
    print(f"relative_pass_rate_lift: {summary.effect_size.relative_pass_rate_lift}")
    print(f"relative_error_reduction: {summary.effect_size.relative_error_reduction}")
    print(
        "bootstrap_delta_ci_pp: "
        f"[{summary.bootstrap_delta.lower * 100:.2f}, {summary.bootstrap_delta.upper * 100:.2f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
