"""Dependency-free GRPO advantage and loss helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any

from codeself.training.common import SequenceTrainingSample, TrainingBatch


@dataclass(frozen=True)
class GRPOLossConfig:
    """Scalar settings for the clipped GRPO objective."""

    clip_epsilon: float = 0.2
    kl_beta: float = 0.05
    max_log_ratio: float = 20.0

    def __post_init__(self) -> None:
        if self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        if self.max_log_ratio <= 0:
            raise ValueError("max_log_ratio must be positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "clip_epsilon": self.clip_epsilon,
            "kl_beta": self.kl_beta,
            "max_log_ratio": self.max_log_ratio,
        }


@dataclass(frozen=True)
class GRPOGroupAdvantage:
    """Group-relative advantages for one task/prompt."""

    task_id: str
    sample_indices: tuple[int, ...]
    rewards: tuple[float, ...]
    advantages: tuple[float, ...]
    mean_reward: float
    reward_std: float

    @property
    def has_reward_variance(self) -> bool:
        return self.reward_std > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "sample_indices": list(self.sample_indices),
            "rewards": list(self.rewards),
            "advantages": list(self.advantages),
            "mean_reward": self.mean_reward,
            "reward_std": self.reward_std,
            "has_reward_variance": self.has_reward_variance,
        }


@dataclass(frozen=True)
class GRPOAdvantageResult:
    """Training batch plus group-relative advantage diagnostics."""

    batch: TrainingBatch
    groups: tuple[GRPOGroupAdvantage, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "batch": self.batch.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class GRPOSampleLoss:
    """Response-token GRPO loss diagnostics for one sequence sample."""

    task_id: str
    sample_index: int
    response_tokens: int
    advantage: float
    policy_loss: float
    kl_loss: float
    total_loss: float
    mean_ratio: float
    mean_clipped_ratio: float
    mean_approx_kl: float
    clipped_tokens: int

    @property
    def clipped_token_fraction(self) -> float:
        return self.clipped_tokens / self.response_tokens if self.response_tokens else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "response_tokens": self.response_tokens,
            "advantage": self.advantage,
            "policy_loss": self.policy_loss,
            "kl_loss": self.kl_loss,
            "total_loss": self.total_loss,
            "mean_ratio": self.mean_ratio,
            "mean_clipped_ratio": self.mean_clipped_ratio,
            "mean_approx_kl": self.mean_approx_kl,
            "clipped_tokens": self.clipped_tokens,
            "clipped_token_fraction": self.clipped_token_fraction,
        }


@dataclass(frozen=True)
class GRPOLossResult:
    """Batch-level GRPO loss summary."""

    config: GRPOLossConfig
    samples: tuple[GRPOSampleLoss, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("GRPOLossResult requires at least one sample")

    @property
    def total_response_tokens(self) -> int:
        return sum(sample.response_tokens for sample in self.samples)

    @property
    def mean_loss(self) -> float:
        return fmean(sample.total_loss for sample in self.samples)

    @property
    def mean_policy_loss(self) -> float:
        return fmean(sample.policy_loss for sample in self.samples)

    @property
    def mean_kl_loss(self) -> float:
        return fmean(sample.kl_loss for sample in self.samples)

    @property
    def mean_approx_kl(self) -> float:
        return fmean(sample.mean_approx_kl for sample in self.samples)

    @property
    def mean_ratio(self) -> float:
        return fmean(sample.mean_ratio for sample in self.samples)

    @property
    def clipped_token_fraction(self) -> float:
        clipped = sum(sample.clipped_tokens for sample in self.samples)
        return clipped / self.total_response_tokens if self.total_response_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "sample_count": len(self.samples),
            "total_response_tokens": self.total_response_tokens,
            "mean_loss": self.mean_loss,
            "mean_policy_loss": self.mean_policy_loss,
            "mean_kl_loss": self.mean_kl_loss,
            "mean_approx_kl": self.mean_approx_kl,
            "mean_ratio": self.mean_ratio,
            "clipped_token_fraction": self.clipped_token_fraction,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def assign_group_relative_advantages(
    batch: TrainingBatch,
    *,
    min_reward_std: float = 1e-8,
) -> GRPOAdvantageResult:
    """Assign normalized group-relative advantages by task ID."""

    if min_reward_std < 0:
        raise ValueError("min_reward_std must be non-negative")

    grouped: dict[str, list[tuple[int, SequenceTrainingSample]]] = {}
    for position, sample in enumerate(batch.samples):
        grouped.setdefault(sample.task_id, []).append((position, sample))

    updated_samples = list(batch.samples)
    groups: list[GRPOGroupAdvantage] = []
    for task_id in sorted(grouped):
        items = sorted(grouped[task_id], key=lambda item: item[1].sample_index)
        rewards = tuple(float(sample.reward) for _, sample in items)
        mean_reward = fmean(rewards)
        reward_std = _population_std(rewards, mean_reward)
        if reward_std > min_reward_std:
            advantages = tuple((reward - mean_reward) / reward_std for reward in rewards)
        else:
            advantages = tuple(0.0 for _ in rewards)
        for (position, sample), advantage in zip(items, advantages, strict=True):
            updated_samples[position] = replace(sample, advantage=advantage)
        groups.append(
            GRPOGroupAdvantage(
                task_id=task_id,
                sample_indices=tuple(sample.sample_index for _, sample in items),
                rewards=rewards,
                advantages=advantages,
                mean_reward=mean_reward,
                reward_std=reward_std,
            )
        )

    return GRPOAdvantageResult(
        batch=TrainingBatch(tuple(updated_samples)),
        groups=tuple(groups),
    )


def compute_grpo_loss(
    batch: TrainingBatch,
    *,
    config: GRPOLossConfig | None = None,
) -> GRPOLossResult:
    """Compute the clipped GRPO surrogate over response tokens only."""

    loss_config = config or GRPOLossConfig()
    return GRPOLossResult(
        config=loss_config,
        samples=tuple(_compute_sample_loss(sample, loss_config) for sample in batch.samples),
    )


def _compute_sample_loss(
    sample: SequenceTrainingSample,
    config: GRPOLossConfig,
) -> GRPOSampleLoss:
    if sample.logprobs is None:
        raise ValueError("GRPO loss requires sample.logprobs")
    if sample.advantage is None:
        raise ValueError("GRPO loss requires sample.advantage")
    logprobs = sample.logprobs
    if logprobs.old_policy_logprobs is None:
        raise ValueError("GRPO loss requires old_policy_logprobs")
    if config.kl_beta > 0 and logprobs.reference_logprobs is None:
        raise ValueError("GRPO KL penalty requires reference_logprobs when kl_beta > 0")

    response_positions = tuple(
        index for index, include in enumerate(sample.masks.response_mask) if include
    )
    if not response_positions:
        raise ValueError("GRPO loss requires at least one response token")

    lower_clip = 1.0 - config.clip_epsilon
    upper_clip = 1.0 + config.clip_epsilon
    policy_losses: list[float] = []
    kl_values: list[float] = []
    ratios: list[float] = []
    clipped_ratios: list[float] = []
    clipped_tokens = 0

    for index in response_positions:
        log_ratio = logprobs.policy_logprobs[index] - logprobs.old_policy_logprobs[index]
        ratio = _exp_clamped(log_ratio, config.max_log_ratio)
        clipped_ratio = min(max(ratio, lower_clip), upper_clip)
        unclipped_objective = ratio * sample.advantage
        clipped_objective = clipped_ratio * sample.advantage
        surrogate_objective = min(unclipped_objective, clipped_objective)
        approx_kl = 0.0
        if logprobs.reference_logprobs is not None:
            reference_delta = logprobs.reference_logprobs[index] - logprobs.policy_logprobs[index]
            safe_delta = _clamp(reference_delta, -config.max_log_ratio, config.max_log_ratio)
            approx_kl = math.exp(safe_delta) - safe_delta - 1.0
        if abs(clipped_ratio - ratio) > 1e-12:
            clipped_tokens += 1
        policy_losses.append(-surrogate_objective)
        kl_values.append(approx_kl)
        ratios.append(ratio)
        clipped_ratios.append(clipped_ratio)

    policy_loss = fmean(policy_losses)
    mean_approx_kl = fmean(kl_values)
    kl_loss = config.kl_beta * mean_approx_kl
    return GRPOSampleLoss(
        task_id=sample.task_id,
        sample_index=sample.sample_index,
        response_tokens=len(response_positions),
        advantage=float(sample.advantage),
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        total_loss=policy_loss + kl_loss,
        mean_ratio=fmean(ratios),
        mean_clipped_ratio=fmean(clipped_ratios),
        mean_approx_kl=mean_approx_kl,
        clipped_tokens=clipped_tokens,
    )


def _population_std(values: tuple[float, ...], mean_value: float) -> float:
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _exp_clamped(value: float, max_abs_value: float) -> float:
    return math.exp(_clamp(value, -max_abs_value, max_abs_value))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
