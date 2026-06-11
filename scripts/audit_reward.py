#!/usr/bin/env python3
"""Run one solution against one task and print the reward breakdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import TaskRegistry  # noqa: E402
from codeself.execution import SandboxedTestRunner  # noqa: E402
from codeself.rewards import CompositeRewardScorer, ConfigurableRewardScorer, RewardModeConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Canonical task JSONL file.")
    parser.add_argument("--task-id", help="Task ID to score. Defaults to the first task.")
    parser.add_argument("--solution-file", type=Path, required=True, help="Python solution file.")
    parser.add_argument("--public-only", action="store_true", help="Do not run hidden tests.")
    parser.add_argument(
        "--reward-mode",
        choices=("correctness_v0", "binary_all_tests_pass", "fractional_pass_rate", "partial_credit"),
        default="correctness_v0",
        help="Reward mode to audit.",
    )
    parser.add_argument("--compile-success-bonus", type=float, default=0.0)
    parser.add_argument("--length-penalty-per-1k-chars", type=float, default=0.0)
    args = parser.parse_args()

    registry = TaskRegistry.from_jsonl(args.tasks)
    task = registry.get(args.task_id) if args.task_id else next(iter(registry))
    solution_code = args.solution_file.read_text(encoding="utf-8")

    result = SandboxedTestRunner().run(task, solution_code, include_hidden=not args.public_only)
    if args.reward_mode == "correctness_v0":
        reward = CompositeRewardScorer().score(result, solution_code=solution_code)
    else:
        reward = ConfigurableRewardScorer(
            RewardModeConfig(
                mode=args.reward_mode,
                name=f"reward_{args.reward_mode}",
                compile_success_bonus=args.compile_success_bonus,
                length_penalty_per_1k_chars=args.length_penalty_per_1k_chars,
            )
        ).score(result, solution_code=solution_code)
    print(json.dumps(reward.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
