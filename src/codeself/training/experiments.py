"""Experiment matrix and selection helpers for GRPO runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class ExperimentVariant:
    """One GRPO experiment variant."""

    name: str
    seed: int
    group_size: int
    kl_beta: float
    reward_name: str = "reward_v0_correctness"
    curriculum_mode: str = "static"
    max_steps: int = 3

    def __post_init__(self) -> None:
        if self.group_size <= 1:
            raise ValueError("group_size must be greater than 1")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.curriculum_mode not in {"static", "informative"}:
            raise ValueError("curriculum_mode must be 'static' or 'informative'")

    @property
    def slug(self) -> str:
        safe_name = self.name.replace("/", "-").replace(" ", "-")
        return (
            f"{safe_name}_seed-{self.seed}_g-{self.group_size}"
            f"_kl-{self.kl_beta:g}_{self.curriculum_mode}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "group_size": self.group_size,
            "kl_beta": self.kl_beta,
            "reward_name": self.reward_name,
            "curriculum_mode": self.curriculum_mode,
            "max_steps": self.max_steps,
            "slug": self.slug,
        }


@dataclass(frozen=True)
class ExperimentRunSummary:
    """Summary for one experiment variant run."""

    variant: ExperimentVariant
    output_dir: str
    steps_completed: int
    mean_reward: float
    informative_prompt_fraction: float
    reward_variance_prompt_fraction: float
    execution_pass_rate: float
    parse_failure_rate: float
    stopped_early: bool

    def score(self, primary_metric: str = "mean_reward") -> float:
        values = self.to_dict()
        if primary_metric not in values:
            raise ValueError(f"unknown primary metric: {primary_metric}")
        return float(values[primary_metric])

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant.to_dict(),
            "output_dir": self.output_dir,
            "steps_completed": self.steps_completed,
            "mean_reward": self.mean_reward,
            "informative_prompt_fraction": self.informative_prompt_fraction,
            "reward_variance_prompt_fraction": self.reward_variance_prompt_fraction,
            "execution_pass_rate": self.execution_pass_rate,
            "parse_failure_rate": self.parse_failure_rate,
            "stopped_early": self.stopped_early,
        }


def build_grpo_experiment_matrix(
    *,
    seeds: tuple[int, ...] = (20260601, 20260602, 20260603),
    group_sizes: tuple[int, ...] = (4, 8),
    kl_betas: tuple[float, ...] = (0.02, 0.05),
    curriculum_modes: tuple[str, ...] = ("static", "informative"),
    reward_names: tuple[str, ...] = ("reward_v0_correctness",),
    max_steps: int = 3,
) -> list[ExperimentVariant]:
    """Build a deterministic GRPO ablation matrix."""

    variants: list[ExperimentVariant] = []
    for reward_name in reward_names:
        for curriculum_mode in curriculum_modes:
            for group_size in group_sizes:
                for kl_beta in kl_betas:
                    for seed in seeds:
                        variants.append(
                            ExperimentVariant(
                                name=f"{reward_name}_{curriculum_mode}",
                                seed=seed,
                                group_size=group_size,
                                kl_beta=kl_beta,
                                reward_name=reward_name,
                                curriculum_mode=curriculum_mode,
                                max_steps=max_steps,
                            )
                        )
    return variants


def select_best_run(
    summaries: list[ExperimentRunSummary],
    *,
    primary_metric: str = "mean_reward",
    require_not_stopped: bool = True,
    minimum_informative_prompt_fraction: float = 0.0,
) -> ExperimentRunSummary:
    """Select the best run without consulting held-out test results."""

    if not 0 <= minimum_informative_prompt_fraction <= 1:
        raise ValueError("minimum_informative_prompt_fraction must be in [0, 1]")
    candidates = [
        summary
        for summary in summaries
        if (not require_not_stopped or not summary.stopped_early)
        and summary.informative_prompt_fraction >= minimum_informative_prompt_fraction
    ]
    if not candidates:
        raise ValueError("no candidate runs available for selection")
    return max(
        candidates,
        key=lambda summary: (
            summary.score(primary_metric),
            summary.informative_prompt_fraction,
            summary.reward_variance_prompt_fraction,
            -summary.parse_failure_rate,
        ),
    )


def aggregate_by_variant_name(summaries: list[ExperimentRunSummary]) -> dict[str, dict[str, float | int]]:
    """Aggregate repeated seeds by variant name/curriculum/group/KL."""

    grouped: dict[str, list[ExperimentRunSummary]] = {}
    for summary in summaries:
        key = (
            f"{summary.variant.name}|g={summary.variant.group_size}|"
            f"kl={summary.variant.kl_beta:g}|{summary.variant.curriculum_mode}"
        )
        grouped.setdefault(key, []).append(summary)

    aggregates: dict[str, dict[str, float | int]] = {}
    for key, values in grouped.items():
        aggregates[key] = {
            "runs": len(values),
            "mean_reward": fmean(summary.mean_reward for summary in values),
            "informative_prompt_fraction": fmean(
                summary.informative_prompt_fraction for summary in values
            ),
            "reward_variance_prompt_fraction": fmean(
                summary.reward_variance_prompt_fraction for summary in values
            ),
            "execution_pass_rate": fmean(summary.execution_pass_rate for summary in values),
            "parse_failure_rate": fmean(summary.parse_failure_rate for summary in values),
        }
    return dict(sorted(aggregates.items()))


def write_experiment_table(
    summaries: list[ExperimentRunSummary],
    path: str | Path,
    *,
    best: ExperimentRunSummary | None = None,
) -> None:
    """Write a Markdown experiment table."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GRPO Experiment Table",
        "",
        "| Selected | Variant | Seed | G | KL | Curriculum | Mean Reward | Informative | Reward Var | Pass Rate | Parse Fail | Stopped |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    best_slug = best.variant.slug if best else None
    for summary in summaries:
        selected = "yes" if summary.variant.slug == best_slug else ""
        lines.append(
            f"| {selected} | {summary.variant.name} | {summary.variant.seed} | "
            f"{summary.variant.group_size} | {summary.variant.kl_beta:g} | "
            f"{summary.variant.curriculum_mode} | {summary.mean_reward:.4f} | "
            f"{summary.informative_prompt_fraction:.4f} | "
            f"{summary.reward_variance_prompt_fraction:.4f} | "
            f"{summary.execution_pass_rate:.4f} | {summary.parse_failure_rate:.4f} | "
            f"{summary.stopped_early} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_experiment_summary_json(
    summaries: list[ExperimentRunSummary],
    path: str | Path,
    *,
    best: ExperimentRunSummary | None = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [summary.to_dict() for summary in summaries],
        "aggregates": aggregate_by_variant_name(summaries),
        "selected": best.to_dict() if best else None,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
