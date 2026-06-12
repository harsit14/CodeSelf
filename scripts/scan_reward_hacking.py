#!/usr/bin/env python3
"""Scan a rollout JSONL file for reward-hacking and degenerate outputs.

Reports the fraction of flagged rollouts and lists examples, so a training run
can be audited for shortcut solutions (hardcoded outputs, constant functions,
echoed expected values, empty/repeated code) before trusting its reward curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import read_rollouts_jsonl  # noqa: E402
from codeself.datasets import TaskRegistry  # noqa: E402
from codeself.rewards import detect_reward_hacking  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True, help="Rollout JSONL to scan.")
    parser.add_argument("--tasks", type=Path, help="Optional task JSONL for hardcoded-answer checks.")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--json-output", type=Path, help="Optional JSON summary output path.")
    parser.add_argument(
        "--fail-on-flag",
        action="store_true",
        help="Exit nonzero if any rollout is flagged (useful in CI).",
    )
    args = parser.parse_args()

    rollouts = read_rollouts_jsonl(args.rollouts)
    tasks_by_id = {}
    if args.tasks:
        tasks_by_id = {task.task_id: task for task in TaskRegistry.from_jsonl(args.tasks)}

    reason_counts: Counter[str] = Counter()
    flagged = []
    for rollout in rollouts:
        task = tasks_by_id.get(rollout.task_id)
        report = detect_reward_hacking(rollout.parsed.code, task)
        if report.flagged:
            for reason in report.reasons:
                reason_counts[reason] += 1
            flagged.append((rollout, report))

    total = len(rollouts)
    flag_rate = len(flagged) / total if total else 0.0
    print(f"scanned {total} rollouts; flagged {len(flagged)} ({flag_rate:.1%})")
    for reason, count in reason_counts.most_common():
        print(f"  {count:4d}  {reason}")

    print()
    for rollout, report in flagged[: args.max_examples]:
        print(f"--- {rollout.task_id} sample {rollout.sample_index} "
              f"(reward {rollout.reward.get('reward')}) ---")
        print(f"    reasons: {'; '.join(report.reasons)}")
        snippet = rollout.parsed.code.strip().splitlines()[:6]
        for line in snippet:
            print(f"    | {line}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "total": total,
            "flagged": len(flagged),
            "flag_rate": flag_rate,
            "reason_counts": dict(reason_counts),
            "examples": [
                {
                    "task_id": rollout.task_id,
                    "sample_index": rollout.sample_index,
                    "reward": rollout.reward.get("reward"),
                    "reasons": list(report.reasons),
                    "code": rollout.parsed.code,
                }
                for rollout, report in flagged[: args.max_examples]
            ],
        }
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.fail_on_flag and flagged:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
