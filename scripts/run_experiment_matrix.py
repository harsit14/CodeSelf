#!/usr/bin/env python3
"""Run a dependency-free GRPO smoke experiment matrix."""

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
    ExperimentRunSummary,
    GRPOSmokeConfig,
    GRPOSmokeTrainer,
    JsonlMetricWriter,
    build_grpo_experiment_matrix,
    select_best_run,
    write_checkpoint_manifest,
    write_experiment_summary_json,
    write_experiment_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Canonical task JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("mock", "static", "transformers"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--static-completion-file", type=Path)
    parser.add_argument("--prompt-template", default="direct_solution_v1")
    parser.add_argument("--split", choices=[split.value for split in Split])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", default="20260601,20260602,20260603")
    parser.add_argument("--group-sizes", default="4,8")
    parser.add_argument("--kl-betas", default="0.02,0.05")
    parser.add_argument("--curriculum-modes", default="static,informative")
    parser.add_argument("--reward-names", default="reward_v0_correctness")
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--stop-if-informative-below", type=float, default=0.0)
    parser.add_argument("--primary-metric", default="mean_reward")
    parser.add_argument("--min-informative-for-selection", type=float, default=0.0)
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
    prompt_template = get_prompt_template(args.prompt_template)
    variants = build_grpo_experiment_matrix(
        seeds=_ints(args.seeds),
        group_sizes=_ints(args.group_sizes),
        kl_betas=_floats(args.kl_betas),
        curriculum_modes=tuple(_strings(args.curriculum_modes)),
        reward_names=tuple(_strings(args.reward_names)),
        max_steps=args.max_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ExperimentRunSummary] = []
    for variant in variants:
        run_dir = args.output_dir / variant.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        selected_tasks = _select_tasks_for_curriculum(tasks, variant.curriculum_mode)
        generator = make_generator(
            args.backend,
            model_name_or_path=args.model,
            static_completion=static_completion,
        )
        config = GRPOSmokeConfig(
            group_size=variant.group_size,
            max_steps=variant.max_steps,
            seed=variant.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            include_hidden=not args.public_only,
            stop_if_informative_prompt_fraction_below=args.stop_if_informative_below,
        )
        result = GRPOSmokeTrainer(
            tasks=selected_tasks,
            generator=generator,
            prompt_template=prompt_template,
            config=config,
        ).run()

        metric_path = run_dir / "metrics.jsonl"
        metric_writer = JsonlMetricWriter(metric_path)
        for metric in result.metrics:
            metric_writer.write(metric.to_dict())
        write_rollouts_jsonl(list(result.last_rollouts), run_dir / "last_rollouts.jsonl")
        checkpoint = {
            **result.checkpoint_payload(),
            "variant": variant.to_dict(),
            "selected_task_count": len(selected_tasks),
        }
        write_checkpoint_manifest(run_dir / "checkpoint_manifest.json", checkpoint)

        last = result.metrics[-1]
        summaries.append(
            ExperimentRunSummary(
                variant=variant,
                output_dir=str(run_dir),
                steps_completed=len(result.metrics),
                mean_reward=last.mean_reward,
                informative_prompt_fraction=last.informative_prompt_fraction,
                reward_variance_prompt_fraction=last.reward_variance_prompt_fraction,
                execution_pass_rate=last.execution_pass_rate,
                parse_failure_rate=last.parse_failure_rate,
                stopped_early=last.stopped_early,
            )
        )

    best = select_best_run(
        summaries,
        primary_metric=args.primary_metric,
        minimum_informative_prompt_fraction=args.min_informative_for_selection,
    )
    write_experiment_table(summaries, args.output_dir / "experiment_table.md", best=best)
    write_experiment_summary_json(summaries, args.output_dir / "summary.json", best=best)
    (args.output_dir / "selected_checkpoint.json").write_text(
        json.dumps(
            {
                "selected_variant": best.variant.to_dict(),
                "selected_output_dir": best.output_dir,
                "primary_metric": args.primary_metric,
                "minimum_informative_prompt_fraction": args.min_informative_for_selection,
                "selection_policy": "dev_or_smoke_metrics_only_no_heldout_test",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"ran {len(summaries)} experiment variants")
    print(f"selected: {best.variant.slug}")
    print(f"selected_mean_reward: {best.mean_reward:.4f}")
    print(f"selected_informative_prompt_fraction: {best.informative_prompt_fraction:.4f}")
    print(f"wrote experiment table to {args.output_dir / 'experiment_table.md'}")
    return 0


def _select_tasks_for_curriculum(tasks, curriculum_mode: str):
    if curriculum_mode == "static":
        return tasks
    # Placeholder for true dynamic sampling: keep deterministic ordering for now.
    return sorted(tasks, key=lambda task: (len(task.hidden_tests) == 0, task.task_id))


def _strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _strings(value))


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _strings(value))


if __name__ == "__main__":
    raise SystemExit(main())
