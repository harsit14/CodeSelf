"""GRPO smoke-test scaffolding.

This module validates the training data path without requiring TRL, GPUs, or
model weights. It computes group-relative rewards and advantage diagnostics over
rollouts, then writes metrics/checkpoint manifests. A real TRL trainer can later
consume the same rollout/reward/evaluation layers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from codeself.agent import CodeGenerator, PromptTemplate, RolloutRecord, generate_rollouts
from codeself.datasets import TaskSpec
from codeself.evaluation import evaluate_rollouts


@dataclass(frozen=True)
class GRPOSmokeConfig:
    """Configuration for a dependency-free GRPO smoke run."""

    group_size: int = 4
    max_steps: int = 3
    seed: int = 20260601
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    include_hidden: bool = True
    eval_every_steps: int = 1
    stop_if_informative_prompt_fraction_below: float = 0.0

    def __post_init__(self) -> None:
        if self.group_size <= 1:
            raise ValueError("group_size must be greater than 1")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.eval_every_steps <= 0:
            raise ValueError("eval_every_steps must be positive")
        if not 0 <= self.stop_if_informative_prompt_fraction_below <= 1:
            raise ValueError("stop threshold must be in [0, 1]")


@dataclass(frozen=True)
class GroupAdvantage:
    """Group-relative reward diagnostics for one prompt/task."""

    task_id: str
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    mean_reward: float
    reward_std: float
    correct_samples: int
    total_samples: int

    @property
    def informative(self) -> bool:
        return 0 < self.correct_samples < self.total_samples

    @property
    def has_reward_variance(self) -> bool:
        return self.reward_std > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rewards": list(self.rewards),
            "advantages": list(self.advantages),
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "correct_samples": self.correct_samples,
            "total_samples": self.total_samples,
            "informative": self.informative,
            "has_reward_variance": self.has_reward_variance,
        }


@dataclass(frozen=True)
class GRPOStepMetrics:
    """Metrics for one smoke training step."""

    step: int
    rollout_count: int
    task_count: int
    mean_reward: float
    reward_std: float
    mean_abs_advantage: float
    informative_prompt_fraction: float
    reward_variance_prompt_fraction: float
    execution_pass_rate: float
    parse_failure_rate: float
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "rollout_count": self.rollout_count,
            "task_count": self.task_count,
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "mean_abs_advantage": self.mean_abs_advantage,
            "informative_prompt_fraction": self.informative_prompt_fraction,
            "reward_variance_prompt_fraction": self.reward_variance_prompt_fraction,
            "execution_pass_rate": self.execution_pass_rate,
            "parse_failure_rate": self.parse_failure_rate,
            "stopped_early": self.stopped_early,
        }


@dataclass(frozen=True)
class GRPOSmokeResult:
    """Result of a smoke training run."""

    metrics: tuple[GRPOStepMetrics, ...]
    last_rollouts: tuple[RolloutRecord, ...]
    groups: tuple[GroupAdvantage, ...]

    def checkpoint_payload(self) -> dict[str, Any]:
        last_step = self.metrics[-1] if self.metrics else None
        return {
            "kind": "grpo_smoke_checkpoint",
            "has_real_model_weights": False,
            "last_step": last_step.to_dict() if last_step else None,
            "group_count": len(self.groups),
            "groups": [group.to_dict() for group in self.groups],
            "notes": (
                "Smoke checkpoint validates rollout/reward/advantage plumbing. "
                "It is not a trained model adapter."
            ),
        }


class GRPOSmokeTrainer:
    """Runs dependency-free GRPO smoke steps."""

    def __init__(
        self,
        *,
        tasks: list[TaskSpec],
        generator: CodeGenerator,
        prompt_template: PromptTemplate,
        config: GRPOSmokeConfig | None = None,
    ) -> None:
        self.tasks = tasks
        self.generator = generator
        self.prompt_template = prompt_template
        self.config = config or GRPOSmokeConfig()

    def run(self) -> GRPOSmokeResult:
        metrics: list[GRPOStepMetrics] = []
        last_rollouts: list[RolloutRecord] = []
        last_groups: list[GroupAdvantage] = []

        for step in range(1, self.config.max_steps + 1):
            rollouts = generate_rollouts(
                self.tasks,
                generator=self.generator,
                prompt_template=self.prompt_template,
                samples_per_task=self.config.group_size,
                seed=self.config.seed + step,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                include_hidden=self.config.include_hidden,
            )
            groups = compute_group_advantages(rollouts)
            step_metrics = summarize_grpo_step(step, rollouts, groups)
            stopped = (
                step_metrics.informative_prompt_fraction
                < self.config.stop_if_informative_prompt_fraction_below
            )
            if stopped:
                step_metrics = GRPOStepMetrics(
                    **{**step_metrics.to_dict(), "stopped_early": True}
                )
            metrics.append(step_metrics)
            last_rollouts = rollouts
            last_groups = groups
            if stopped:
                break

        return GRPOSmokeResult(
            metrics=tuple(metrics),
            last_rollouts=tuple(last_rollouts),
            groups=tuple(last_groups),
        )


def compute_group_advantages(records: list[RolloutRecord]) -> list[GroupAdvantage]:
    """Compute group-relative normalized advantages by task ID."""

    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)

    groups: list[GroupAdvantage] = []
    for task_id in sorted(grouped):
        task_records = sorted(grouped[task_id], key=lambda record: record.sample_index)
        rewards = tuple(float(record.reward.get("reward", 0.0)) for record in task_records)
        mean_reward = fmean(rewards) if rewards else 0.0
        reward_std = _population_std(rewards, mean_reward)
        if reward_std > 0:
            advantages = tuple((reward - mean_reward) / reward_std for reward in rewards)
        else:
            advantages = tuple(0.0 for _ in rewards)
        correct_samples = sum(1 for record in task_records if bool(record.execution.get("passed")))
        groups.append(
            GroupAdvantage(
                task_id=task_id,
                rewards=rewards,
                advantages=advantages,
                mean_reward=mean_reward,
                reward_std=reward_std,
                correct_samples=correct_samples,
                total_samples=len(task_records),
            )
        )
    return groups


def summarize_grpo_step(
    step: int,
    records: list[RolloutRecord],
    groups: list[GroupAdvantage],
) -> GRPOStepMetrics:
    """Summarize rollout and advantage metrics for a smoke step."""

    evaluation = evaluate_rollouts(records, ks=(1,))
    rewards = [float(record.reward.get("reward", 0.0)) for record in records]
    mean_reward = fmean(rewards) if rewards else 0.0
    reward_std = _population_std(tuple(rewards), mean_reward)
    advantages = [abs(value) for group in groups for value in group.advantages]
    return GRPOStepMetrics(
        step=step,
        rollout_count=len(records),
        task_count=len(groups),
        mean_reward=mean_reward,
        reward_std=reward_std,
        mean_abs_advantage=fmean(advantages) if advantages else 0.0,
        informative_prompt_fraction=_safe_rate(
            sum(1 for group in groups if group.informative),
            len(groups),
        ),
        reward_variance_prompt_fraction=_safe_rate(
            sum(1 for group in groups if group.has_reward_variance),
            len(groups),
        ),
        execution_pass_rate=evaluation.execution_pass_rate,
        parse_failure_rate=evaluation.parse_failure_rate,
    )


def _population_std(values: tuple[float, ...], mean_value: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
