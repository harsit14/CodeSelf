#!/usr/bin/env python3
"""Run dependency-free PPO smoke training diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import get_prompt_template, make_generator, write_rollouts_jsonl  # noqa: E402
from codeself.config import config_get, load_config_file  # noqa: E402
from codeself.datasets import Split, TaskRegistry, check_dataset_quality  # noqa: E402
from codeself.rewards import make_reward_scorer_from_mode  # noqa: E402
from codeself.training import (  # noqa: E402
    JsonlMetricWriter,
    PPOSmokeConfig,
    PPOSmokeTrainer,
    write_checkpoint_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON/simple-YAML PPO smoke config.")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--backend", choices=("mock", "static", "transformers"))
    parser.add_argument("--model")
    parser.add_argument("--static-completion-file", type=Path)
    parser.add_argument("--prompt-template")
    parser.add_argument("--split", choices=[split.value for split in Split])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--public-only", action="store_true")
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--value-loss-coef", type=float)
    parser.add_argument(
        "--reward-mode",
        choices=("correctness_v0", "binary_all_tests_pass", "fractional_pass_rate", "partial_credit"),
    )
    parser.add_argument("--compile-success-bonus", type=float)
    parser.add_argument("--length-penalty-per-1k-chars", type=float)
    parser.add_argument("--skip-data-quality-checks", action="store_true", default=None)
    parser.add_argument("--near-duplicate-threshold", type=float)
    args = parser.parse_args()

    config_doc = load_config_file(args.config) if args.config else {}
    task_path = _path_value(args.tasks, config_get(config_doc, "data.tasks"))
    output_dir = _path_value(args.output_dir, config_get(config_doc, "output.dir"))
    if task_path is None:
        raise SystemExit("--tasks is required unless provided by --config data.tasks")
    if output_dir is None:
        raise SystemExit("--output-dir is required unless provided by --config output.dir")

    registry = TaskRegistry.from_jsonl(task_path)
    all_tasks = list(registry)
    if bool(
        _value(
            args.skip_data_quality_checks,
            config_get(config_doc, "data.skip_quality_checks"),
            default=False,
        )
    ):
        quality = None
    else:
        quality = check_dataset_quality(
            all_tasks,
            near_duplicate_threshold=float(
                _value(
                    args.near_duplicate_threshold,
                    config_get(config_doc, "data.near_duplicate_threshold"),
                    default=0.92,
                )
            ),
        )
        if not quality.passed:
            _print_quality_failures(quality)
            return 1

    split = _value(args.split, config_get(config_doc, "data.split"), default=None)
    tasks = registry.by_split(split) if split else all_tasks
    limit = _value(args.limit, config_get(config_doc, "data.limit"), default=None)
    if limit is not None:
        tasks = tasks[: int(limit)]
    if not tasks:
        raise SystemExit("no tasks selected")

    static_completion_path = _path_value(
        args.static_completion_file,
        config_get(config_doc, "generation.static_completion_file"),
    )
    static_completion = (
        static_completion_path.read_text(encoding="utf-8") if static_completion_path else None
    )
    backend = str(_value(args.backend, config_get(config_doc, "generation.backend"), default="mock"))
    model = _value(args.model, config_get(config_doc, "generation.model"), default=None)
    generator = make_generator(
        backend,
        model_name_or_path=model,
        static_completion=static_completion,
    )
    reward_mode = str(
        _value(args.reward_mode, config_get(config_doc, "reward.mode"), default="correctness_v0")
    )
    scorer = make_reward_scorer_from_mode(
        reward_mode,
        compile_success_bonus=float(
            _value(
                args.compile_success_bonus,
                config_get(config_doc, "reward.compile_success_bonus"),
                default=0.0,
            )
        ),
        length_penalty_per_1k_chars=float(
            _value(
                args.length_penalty_per_1k_chars,
                config_get(config_doc, "reward.length_penalty_per_1k_chars"),
                default=0.0,
            )
        ),
    )
    include_hidden = bool(
        _value(
            False if args.public_only else None,
            config_get(config_doc, "execution.include_hidden"),
            default=True,
        )
    )
    config = PPOSmokeConfig(
        samples_per_task=int(
            _value(
                args.samples_per_task,
                config_get(config_doc, "generation.samples_per_task"),
                default=4,
            )
        ),
        max_steps=int(_value(args.max_steps, config_get(config_doc, "training.max_steps"), default=3)),
        seed=int(_value(args.seed, config_get(config_doc, "experiment.seed"), default=20260601)),
        max_new_tokens=int(
            _value(args.max_new_tokens, config_get(config_doc, "generation.max_new_tokens"), default=512)
        ),
        temperature=float(
            _value(args.temperature, config_get(config_doc, "generation.temperature"), default=0.8)
        ),
        top_p=float(_value(args.top_p, config_get(config_doc, "generation.top_p"), default=0.95)),
        include_hidden=include_hidden,
        reward_mode=reward_mode,
        clip_epsilon=float(_value(args.clip_epsilon, config_get(config_doc, "training.clip_epsilon"), default=0.2)),
        value_loss_coef=float(
            _value(args.value_loss_coef, config_get(config_doc, "training.value_loss_coef"), default=0.5)
        ),
    )
    result = PPOSmokeTrainer(
        tasks=tasks,
        generator=generator,
        prompt_template=get_prompt_template(
            str(_value(args.prompt_template, config_get(config_doc, "prompt.template"), default="direct_solution_v1"))
        ),
        config=config,
        scorer=scorer,
    ).run()

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "metrics.jsonl"
    writer = JsonlMetricWriter(metric_path)
    for metric in result.metrics:
        writer.write(metric.to_dict())
    write_rollouts_jsonl(list(result.last_rollouts), output_dir / "last_rollouts.jsonl")
    checkpoint = {
        **result.checkpoint_payload(),
        "config": {
            "backend": backend,
            "prompt_template": str(
                _value(args.prompt_template, config_get(config_doc, "prompt.template"), default="direct_solution_v1")
            ),
            "config_path": str(args.config) if args.config else "",
            "task_count": len(tasks),
            **config.__dict__,
        },
    }
    write_checkpoint_manifest(output_dir / "checkpoint_manifest.json", checkpoint)
    (output_dir / "summary.json").write_text(
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
    print(f"wrote PPO smoke outputs to {output_dir}")
    print(f"steps_completed: {len(result.metrics)}")
    print(f"rollout_count: {last.rollout_count}")
    print(f"mean_reward: {last.mean_reward:.4f}")
    print(f"clipped_fraction: {last.clipped_fraction:.4f}")
    print(f"value_loss: {last.value_loss:.4f}")
    return 0


def _value(cli_value: Any, config_value: Any, *, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _path_value(cli_value: Path | None, config_value: Any) -> Path | None:
    if cli_value is not None:
        return cli_value
    if config_value is None:
        return None
    return Path(str(config_value))


def _print_quality_failures(quality) -> None:
    print("dataset quality checks failed")
    for leak in quality.hidden_leaks:
        print(f"hidden leak: {leak.task_id}:{leak.test_name} in {leak.surface}")
    for finding in quality.contamination_findings:
        print(
            f"contamination: {finding.kind} {finding.task_id_a} ({finding.split_a}) <> "
            f"{finding.task_id_b} ({finding.split_b}), score={finding.score:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
