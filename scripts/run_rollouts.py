#!/usr/bin/env python3
"""Generate, parse, execute, score, and store rollout records."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    generate_rollouts,
    get_prompt_template,
    make_generator,
    write_rollouts_jsonl,
)
from codeself.config import config_get, load_config_file  # noqa: E402
from codeself.datasets import Split, TaskRegistry, check_dataset_quality  # noqa: E402
from codeself.rewards import make_reward_scorer_from_mode  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional JSON/simple-YAML rollout config.")
    parser.add_argument("--tasks", type=Path, help="Canonical task JSONL file.")
    parser.add_argument("--output", type=Path, help="Output rollout JSONL file.")
    parser.add_argument("--backend", choices=("mock", "static", "transformers"))
    parser.add_argument("--model", help="Local model path/name for transformers backend.")
    parser.add_argument("--static-completion-file", type=Path, help="Completion file for static backend.")
    parser.add_argument("--prompt-template")
    parser.add_argument("--split", choices=[split.value for split in Split], help="Optional split filter.")
    parser.add_argument("--limit", type=int, help="Optional task limit after split filtering.")
    parser.add_argument("--samples-per-task", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--public-only", action="store_true", help="Skip hidden tests.")
    parser.add_argument(
        "--reward-mode",
        choices=("correctness_v0", "binary_all_tests_pass", "fractional_pass_rate", "partial_credit"),
        help="Reward mode for rollout scoring.",
    )
    parser.add_argument("--compile-success-bonus", type=float)
    parser.add_argument("--length-penalty-per-1k-chars", type=float)
    parser.add_argument("--skip-data-quality-checks", action="store_true", default=None)
    parser.add_argument("--near-duplicate-threshold", type=float)
    args = parser.parse_args()

    config = load_config_file(args.config) if args.config else {}
    task_path = _path_value(args.tasks, config_get(config, "data.tasks"))
    output_path = _path_value(args.output, config_get(config, "output.rollouts"))
    if task_path is None:
        raise SystemExit("--tasks is required unless provided by --config data.tasks")
    if output_path is None:
        raise SystemExit("--output is required unless provided by --config output.rollouts")

    registry = TaskRegistry.from_jsonl(task_path)
    all_tasks = list(registry)
    quality_checks_enabled = not _value(
        args.skip_data_quality_checks,
        config_get(config, "data.skip_quality_checks", False),
        default=False,
    )
    near_duplicate_threshold = float(
        _value(
            args.near_duplicate_threshold,
            config_get(config, "data.near_duplicate_threshold"),
            default=0.92,
        )
    )
    if quality_checks_enabled:
        quality = check_dataset_quality(
            all_tasks,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        if not quality.passed:
            _print_quality_failures(quality)
            return 1

    split = _value(args.split, config_get(config, "data.split"), default=None)
    tasks = registry.by_split(split) if split else all_tasks
    limit = _value(args.limit, config_get(config, "data.limit"), default=None)
    if limit is not None:
        tasks = tasks[: int(limit)]

    static_completion = None
    static_completion_path = _path_value(
        args.static_completion_file,
        config_get(config, "generation.static_completion_file"),
    )
    if static_completion_path:
        static_completion = static_completion_path.read_text(encoding="utf-8")

    backend = str(_value(args.backend, config_get(config, "generation.backend"), default="mock"))
    model = _value(args.model, config_get(config, "generation.model"), default=None)
    generator = make_generator(
        backend,
        model_name_or_path=model,
        static_completion=static_completion,
    )
    prompt_template_name = str(
        _value(args.prompt_template, config_get(config, "prompt.template"), default="direct_solution_v1")
    )
    prompt_template = get_prompt_template(prompt_template_name)
    reward_mode = str(_value(args.reward_mode, config_get(config, "reward.mode"), default="correctness_v0"))
    scorer = _make_scorer(
        reward_mode,
        compile_success_bonus=float(
            _value(
                args.compile_success_bonus,
                config_get(config, "reward.compile_success_bonus"),
                default=0.0,
            )
        ),
        length_penalty_per_1k_chars=float(
            _value(
                args.length_penalty_per_1k_chars,
                config_get(config, "reward.length_penalty_per_1k_chars"),
                default=0.0,
            )
        ),
    )
    include_hidden = bool(
        _value(
            False if args.public_only else None,
            config_get(config, "execution.include_hidden"),
            default=True,
        )
    )
    seed = int(_value(args.seed, config_get(config, "experiment.seed"), default=20260601))
    records = generate_rollouts(
        tasks,
        generator=generator,
        prompt_template=prompt_template,
        samples_per_task=int(
            _value(args.samples_per_task, config_get(config, "generation.samples_per_task"), default=1)
        ),
        seed=seed,
        max_new_tokens=int(
            _value(args.max_new_tokens, config_get(config, "generation.max_new_tokens"), default=512)
        ),
        temperature=float(
            _value(args.temperature, config_get(config, "generation.temperature"), default=0.8)
        ),
        top_p=float(_value(args.top_p, config_get(config, "generation.top_p"), default=0.95)),
        include_hidden=include_hidden,
        scorer=scorer,
        metadata={
            "reward_mode": reward_mode,
            "config_path": str(args.config) if args.config else "",
        },
    )
    write_rollouts_jsonl(records, output_path)
    _print_summary(records, output_path)
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


def _make_scorer(
    reward_mode: str,
    *,
    compile_success_bonus: float,
    length_penalty_per_1k_chars: float,
):
    return make_reward_scorer_from_mode(
        reward_mode,
        compile_success_bonus=compile_success_bonus,
        length_penalty_per_1k_chars=length_penalty_per_1k_chars,
    )


def _path_value(cli_value: Path | None, config_value: Any) -> Path | None:
    if cli_value is not None:
        return cli_value
    if config_value is None:
        return None
    return Path(str(config_value))


def _value(cli_value: Any, config_value: Any, *, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


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
