"""PPO smoke-test scaffolding.

This module does not update model weights. It computes PPO-style diagnostics
over rollout batches so PPO and GRPO runs can be compared before installing
full training dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from codeself.agent import CodeGenerator, PromptTemplate, RolloutRecord, generate_rollouts
from codeself.datasets import TaskSpec
from codeself.evaluation import evaluate_rollouts
from codeself.rewards import RewardScorer


@dataclass(frozen=True)
class PPOSmokeConfig:
    """Configuration for dependency-free PPO smoke diagnostics."""

    samples_per_task: int = 4
    max_steps: int = 3
    seed: int = 20260601
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    include_hidden: bool = True
    reward_mode: str = "correctness_v0"
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5

    def __post_init__(self) -> None:
        if self.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")
        if self.value_loss_coef < 0:
            raise ValueError("value_loss_coef must be non-negative")


@dataclass(frozen=True)
class PPOStepMetrics:
    """PPO-style diagnostics for one smoke step."""

    step: int
    rollout_count: int
    task_count: int
    mean_reward: float
    reward_std: float
    value_baseline: float
    advantage_mean: float
    advantage_std: float
    mean_abs_advantage: float
    clipped_fraction: float
    policy_objective: float
    value_loss: float
    execution_pass_rate: float
    parse_failure_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "rollout_count": self.rollout_count,
            "task_count": self.task_count,
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "value_baseline": self.value_baseline,
            "advantage_mean": self.advantage_mean,
            "advantage_std": self.advantage_std,
            "mean_abs_advantage": self.mean_abs_advantage,
            "clipped_fraction": self.clipped_fraction,
            "policy_objective": self.policy_objective,
            "value_loss": self.value_loss,
            "execution_pass_rate": self.execution_pass_rate,
            "parse_failure_rate": self.parse_failure_rate,
        }


@dataclass(frozen=True)
class PPOSmokeResult:
    """Result of a PPO smoke run."""

    metrics: tuple[PPOStepMetrics, ...]
    last_rollouts: tuple[RolloutRecord, ...]

    def checkpoint_payload(self) -> dict[str, Any]:
        last_step = self.metrics[-1] if self.metrics else None
        return {
            "kind": "ppo_smoke_checkpoint",
            "has_real_model_weights": False,
            "last_step": last_step.to_dict() if last_step else None,
            "notes": (
                "Smoke checkpoint validates PPO-style rollout/reward/advantage plumbing. "
                "It is not a trained model adapter."
            ),
        }


class PPOSmokeTrainer:
    """Runs dependency-free PPO smoke steps."""

    def __init__(
        self,
        *,
        tasks: list[TaskSpec],
        generator: CodeGenerator,
        prompt_template: PromptTemplate,
        config: PPOSmokeConfig | None = None,
        scorer: RewardScorer | None = None,
    ) -> None:
        self.tasks = tasks
        self.generator = generator
        self.prompt_template = prompt_template
        self.config = config or PPOSmokeConfig()
        self.scorer = scorer

    def run(self) -> PPOSmokeResult:
        metrics: list[PPOStepMetrics] = []
        last_rollouts: list[RolloutRecord] = []
        for step in range(1, self.config.max_steps + 1):
            rollouts = generate_rollouts(
                self.tasks,
                generator=self.generator,
                prompt_template=self.prompt_template,
                samples_per_task=self.config.samples_per_task,
                seed=self.config.seed + step,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                include_hidden=self.config.include_hidden,
                scorer=self.scorer,
                metadata={"reward_mode": self.config.reward_mode},
            )
            metrics.append(summarize_ppo_step(step, rollouts, self.config))
            last_rollouts = rollouts
        return PPOSmokeResult(metrics=tuple(metrics), last_rollouts=tuple(last_rollouts))


def summarize_ppo_step(
    step: int,
    records: list[RolloutRecord],
    config: PPOSmokeConfig | None = None,
) -> PPOStepMetrics:
    """Summarize PPO-style diagnostics for a rollout batch."""

    ppo_config = config or PPOSmokeConfig()
    evaluation = evaluate_rollouts(records, ks=(1,))
    rewards = tuple(float(record.reward.get("reward", 0.0)) for record in records)
    mean_reward = fmean(rewards) if rewards else 0.0
    reward_std = _population_std(rewards, mean_reward)
    value_baseline = mean_reward
    advantages = tuple(reward - value_baseline for reward in rewards)
    advantage_mean = fmean(advantages) if advantages else 0.0
    advantage_std = _population_std(advantages, advantage_mean)
    ratios = _simulated_importance_ratios(records)
    clipped_ratios = tuple(
        max(1 - ppo_config.clip_epsilon, min(1 + ppo_config.clip_epsilon, ratio))
        for ratio in ratios
    )
    clipped_terms = tuple(
        min(ratio * advantage, clipped * advantage)
        for ratio, clipped, advantage in zip(ratios, clipped_ratios, advantages, strict=False)
    )
    value_losses = tuple((reward - value_baseline) ** 2 for reward in rewards)
    return PPOStepMetrics(
        step=step,
        rollout_count=len(records),
        task_count=len({record.task_id for record in records}),
        mean_reward=mean_reward,
        reward_std=reward_std,
        value_baseline=value_baseline,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
        mean_abs_advantage=fmean(abs(value) for value in advantages) if advantages else 0.0,
        clipped_fraction=_safe_rate(
            sum(1 for ratio, clipped in zip(ratios, clipped_ratios, strict=False) if ratio != clipped),
            len(ratios),
        ),
        policy_objective=fmean(clipped_terms) if clipped_terms else 0.0,
        value_loss=fmean(value_losses) * ppo_config.value_loss_coef if value_losses else 0.0,
        execution_pass_rate=evaluation.execution_pass_rate,
        parse_failure_rate=evaluation.parse_failure_rate,
    )


def _simulated_importance_ratios(records: list[RolloutRecord]) -> tuple[float, ...]:
    """Deterministic placeholder ratios for smoke diagnostics.

    Real PPO computes policy/reference probability ratios. The smoke runner does
    not have logprobs, so this provides stable nontrivial values for plumbing,
    clipping, and reporting tests.
    """

    ratios: list[float] = []
    for record in records:
        bucket = (record.sample_index % 5) - 2
        ratios.append(1.0 + bucket * 0.08)
    return tuple(ratios)


def _population_std(values: tuple[float, ...], mean_value: float) -> float:
    if not values:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
