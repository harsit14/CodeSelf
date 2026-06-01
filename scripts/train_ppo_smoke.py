#!/usr/bin/env python3
"""Run dependency-free PPO smoke training diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import get_prompt_template, make_generator, write_rollouts_jsonl  # noqa: E402
from codeself.datasets import Split, TaskRegistry  # noqa: E402
from codeself.training import (  # noqa: E402
    JsonlMetricWriter,
    PPOSmokeConfig,
    PPOSmokeTrainer,
    write_checkpoint_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("mock", "static", "transformers"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--static-completion-file", type=Path)
    parser.add_argument("--prompt-template", default="direct_solution_v1")
    parser.add_argument("--split", choices=[split.value for split in Split])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    args = parser.parse_args()

    registry = TaskRegistry.from_jsonl(args.tasks)
    tasks = registry.by_split(args.split) if args.split else list(registry)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    static_completion = (
        args.static_completion_file.read_text(encoding="utf-8")
        if args.static_completion_file
        else None
    )
    generator = make_generator(
        args.backend,
        model_name_or_path=args.model,
        static_completion=static_completion,
    )
    config = PPOSmokeConfig(
        samples_per_task=args.samples_per_task,
        max_steps=args.max_steps,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        include_hidden=not args.public_only,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
    )
    result = PPOSmokeTrainer(
        tasks=tasks,
        generator=generator,
        prompt_template=get_prompt_template(args.prompt_template),
        config=config,
    ).run()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = args.output_dir / "metrics.jsonl"
    writer = JsonlMetricWriter(metric_path)
    for metric in result.metrics:
        writer.write(metric.to_dict())
    write_rollouts_jsonl(list(result.last_rollouts), args.output_dir / "last_rollouts.jsonl")
    checkpoint = {
        **result.checkpoint_payload(),
        "config": {
            "backend": args.backend,
            "prompt_template": args.prompt_template,
            "task_count": len(tasks),
            **config.__dict__,
        },
    }
    write_checkpoint_manifest(args.output_dir / "checkpoint_manifest.json", checkpoint)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "metrics_path": str(metric_path),
                "steps": [metric.to_dict() for metric in result.metrics],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    last = result.metrics[-1]
    print(f"wrote PPO smoke outputs to {args.output_dir}")
    print(f"steps_completed: {len(result.metrics)}")
    print(f"rollout_count: {last.rollout_count}")
    print(f"mean_reward: {last.mean_reward:.4f}")
    print(f"clipped_fraction: {last.clipped_fraction:.4f}")
    print(f"value_loss: {last.value_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
