#!/usr/bin/env python3
"""Generate, parse, execute, score, and store rollout records."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    generate_rollouts,
    get_prompt_template,
    make_generator,
    write_rollouts_jsonl,
)
from codeself.datasets import Split, TaskRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Canonical task JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="Output rollout JSONL file.")
    parser.add_argument("--backend", choices=("mock", "static", "transformers"), default="mock")
    parser.add_argument("--model", help="Local model path/name for transformers backend.")
    parser.add_argument("--static-completion-file", type=Path, help="Completion file for static backend.")
    parser.add_argument("--prompt-template", default="direct_solution_v1")
    parser.add_argument("--split", choices=[split.value for split in Split], help="Optional split filter.")
    parser.add_argument("--limit", type=int, help="Optional task limit after split filtering.")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--public-only", action="store_true", help="Skip hidden tests.")
    args = parser.parse_args()

    registry = TaskRegistry.from_jsonl(args.tasks)
    tasks = registry.by_split(args.split) if args.split else list(registry)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    static_completion = None
    if args.static_completion_file:
        static_completion = args.static_completion_file.read_text(encoding="utf-8")

    generator = make_generator(
        args.backend,
        model_name_or_path=args.model,
        static_completion=static_completion,
    )
    prompt_template = get_prompt_template(args.prompt_template)
    records = generate_rollouts(
        tasks,
        generator=generator,
        prompt_template=prompt_template,
        samples_per_task=args.samples_per_task,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_hidden=not args.public_only,
    )
    write_rollouts_jsonl(records, args.output)
    _print_summary(records, args.output)
    return 0


def _print_summary(records, output: Path) -> None:
    rewards = [float(record.reward["reward"]) for record in records]
    parse_failures = [record for record in records if record.parsed.status.value != "ok"]
    passed = [record for record in records if record.execution.get("passed")]
    print(f"wrote {len(records)} rollout records to {output}")
    if records:
        print(f"parse_failure_rate: {len(parse_failures) / len(records):.4f}")
        print(f"pass_rate: {len(passed) / len(records):.4f}")
        print(f"mean_reward: {statistics.fmean(rewards):.4f}")
        print(f"min_reward: {min(rewards):.4f}")
        print(f"max_reward: {max(rewards):.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
