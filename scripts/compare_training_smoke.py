#!/usr/bin/env python3
"""Compare final GRPO and PPO smoke metric files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import compare_algorithm_metrics, read_last_metric, write_algorithm_comparison  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grpo-metrics", type=Path, required=True)
    parser.add_argument("--ppo-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = compare_algorithm_metrics(
        read_last_metric(args.grpo_metrics),
        read_last_metric(args.ppo_metrics),
    )
    write_algorithm_comparison(comparison, args.output)
    print(f"wrote PPO/GRPO comparison to {args.output}")
    print(f"rollout_budget_matched: {comparison.rollout_budget_matched}")
    print(f"mean_reward_delta_grpo_minus_ppo: {comparison.mean_reward_delta_grpo_minus_ppo:.4f}")
    print(f"pass_rate_delta_grpo_minus_ppo: {comparison.pass_rate_delta_grpo_minus_ppo:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
