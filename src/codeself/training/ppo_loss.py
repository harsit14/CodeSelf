"""Dependency-free PPO advantage and loss helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from codeself.training.common import SequenceTrainingSample, TrainingBatch


@dataclass(frozen=True)
class PPOLossConfig:
    """Scalar settings for the clipped PPO objective."""

    clip_epsilon: float = 0.2
    value_clip_epsilon: float | None = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    kl_beta: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 1.0
    normalize_advantages: bool = True
    max_log_ratio: float = 20.0

    def __post_init__(self) -> None:
        if self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")
        if self.value_clip_epsilon is not None and self.value_clip_epsilon <= 0:
            raise ValueError("value_clip_epsilon must be positive when set")
        if self.value_loss_coef < 0:
            raise ValueError("value_loss_coef must be non-negative")
        if self.entropy_coef < 0:
            raise ValueError("entropy_coef must be non-negative")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.max_log_ratio <= 0:
            raise ValueError("max_log_ratio must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "clip_epsilon": self.clip_epsilon,
            "value_clip_epsilon": self.value_clip_epsilon,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "kl_beta": self.kl_beta,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "normalize_advantages": self.normalize_advantages,
            "max_log_ratio": self.max_log_ratio,
        }


@dataclass(frozen=True)
class PPOValueEstimate:
    """Current and rollout value-head estimates aligned to a sample sequence."""

    values: tuple[float, ...]
    old_values: tuple[float, ...] | None = None
    rewards: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "values": list(self.values),
            "old_values": list(self.old_values) if self.old_values is not None else None,
            "rewards": list(self.rewards) if self.rewards is not None else None,
        }


@dataclass(frozen=True)
class PPOValueTarget:
    """GAE-derived advantages and value targets for one sample."""

    task_id: str
    sample_index: int
    response_positions: tuple[int, ...]
    rewards: tuple[float, ...]
    values: tuple[float, ...]
    old_values: tuple[float, ...]
    raw_advantages: tuple[float, ...]
    advantages: tuple[float, ...]
    returns: tuple[float, ...]

    @property
    def response_tokens(self) -> int:
        return len(self.response_positions)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "response_positions": list(self.response_positions),
            "rewards": list(self.rewards),
            "values": list(self.values),
            "old_values": list(self.old_values),
            "raw_advantages": list(self.raw_advantages),
            "advantages": list(self.advantages),
            "returns": list(self.returns),
            "response_tokens": self.response_tokens,
        }


@dataclass(frozen=True)
class PPOValueTargetResult:
    """Value targets for a batch plus advantage diagnostics."""

    config: PPOLossConfig
    targets: tuple[PPOValueTarget, ...]

    @property
    def total_response_tokens(self) -> int:
        return sum(target.response_tokens for target in self.targets)

    @property
    def mean_advantage(self) -> float:
        values = tuple(value for target in self.targets for value in target.advantages)
        return fmean(values) if values else 0.0

    @property
    def mean_return(self) -> float:
        values = tuple(value for target in self.targets for value in target.returns)
        return fmean(values) if values else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "total_response_tokens": self.total_response_tokens,
            "mean_advantage": self.mean_advantage,
            "mean_return": self.mean_return,
            "targets": [target.to_dict() for target in self.targets],
        }


@dataclass(frozen=True)
class PPOSampleLoss:
    """Response-token PPO loss diagnostics for one sample."""

    task_id: str
    sample_index: int
    response_tokens: int
    policy_loss: float
    value_loss: float
    entropy_loss: float
    kl_loss: float
    total_loss: float
    mean_ratio: float
    mean_clipped_ratio: float
    mean_advantage: float
    mean_return: float
    mean_entropy: float
    mean_approx_kl: float
    clipped_policy_tokens: int
    clipped_value_tokens: int

    @property
    def clipped_policy_fraction(self) -> float:
        return self.clipped_policy_tokens / self.response_tokens if self.response_tokens else 0.0

    @property
    def clipped_value_fraction(self) -> float:
        return self.clipped_value_tokens / self.response_tokens if self.response_tokens else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "response_tokens": self.response_tokens,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy_loss": self.entropy_loss,
            "kl_loss": self.kl_loss,
            "total_loss": self.total_loss,
            "mean_ratio": self.mean_ratio,
            "mean_clipped_ratio": self.mean_clipped_ratio,
            "mean_advantage": self.mean_advantage,
            "mean_return": self.mean_return,
            "mean_entropy": self.mean_entropy,
            "mean_approx_kl": self.mean_approx_kl,
            "clipped_policy_tokens": self.clipped_policy_tokens,
            "clipped_policy_fraction": self.clipped_policy_fraction,
            "clipped_value_tokens": self.clipped_value_tokens,
            "clipped_value_fraction": self.clipped_value_fraction,
        }


@dataclass(frozen=True)
class PPOLossResult:
    """Batch-level PPO loss summary."""

    config: PPOLossConfig
    value_targets: PPOValueTargetResult
    samples: tuple[PPOSampleLoss, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("PPOLossResult requires at least one sample")

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
    def mean_value_loss(self) -> float:
        return fmean(sample.value_loss for sample in self.samples)

    @property
    def mean_entropy_loss(self) -> float:
        return fmean(sample.entropy_loss for sample in self.samples)

    @property
    def mean_kl_loss(self) -> float:
        return fmean(sample.kl_loss for sample in self.samples)

    @property
    def clipped_policy_fraction(self) -> float:
        clipped = sum(sample.clipped_policy_tokens for sample in self.samples)
        return clipped / self.total_response_tokens if self.total_response_tokens else 0.0

    @property
    def clipped_value_fraction(self) -> float:
        clipped = sum(sample.clipped_value_tokens for sample in self.samples)
        return clipped / self.total_response_tokens if self.total_response_tokens else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "sample_count": len(self.samples),
            "total_response_tokens": self.total_response_tokens,
            "mean_loss": self.mean_loss,
            "mean_policy_loss": self.mean_policy_loss,
            "mean_value_loss": self.mean_value_loss,
            "mean_entropy_loss": self.mean_entropy_loss,
            "mean_kl_loss": self.mean_kl_loss,
            "clipped_policy_fraction": self.clipped_policy_fraction,
            "clipped_value_fraction": self.clipped_value_fraction,
            "value_targets": self.value_targets.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
        }


ValueEstimateInput = Mapping[tuple[str, int], PPOValueEstimate] | Sequence[PPOValueEstimate]


def prepare_ppo_value_targets(
    batch: TrainingBatch,
    value_estimates: ValueEstimateInput,
    *,
    config: PPOLossConfig | None = None,
) -> PPOValueTargetResult:
    """Compute GAE advantages and value targets over response tokens."""

    loss_config = config or PPOLossConfig()
    targets = tuple(
        _prepare_sample_target(
            sample,
            _estimate_for(position, sample, value_estimates),
            loss_config,
        )
        for position, sample in enumerate(batch.samples)
    )
    if loss_config.normalize_advantages:
        targets = _normalize_target_advantages(targets)
    return PPOValueTargetResult(config=loss_config, targets=targets)


def compute_ppo_loss(
    batch: TrainingBatch,
    value_estimates: ValueEstimateInput,
    *,
    config: PPOLossConfig | None = None,
) -> PPOLossResult:
    """Compute the clipped PPO objective over response tokens only."""

    loss_config = config or PPOLossConfig()
    value_targets = prepare_ppo_value_targets(batch, value_estimates, config=loss_config)
    samples = tuple(
        _compute_sample_loss(sample, target, loss_config)
        for sample, target in zip(batch.samples, value_targets.targets, strict=True)
    )
    return PPOLossResult(config=loss_config, value_targets=value_targets, samples=samples)


def _prepare_sample_target(
    sample: SequenceTrainingSample,
    estimate: PPOValueEstimate,
    config: PPOLossConfig,
) -> PPOValueTarget:
    response_positions = _response_positions(sample)
    if not response_positions:
        raise ValueError("PPO value targets require at least one response token")
    _validate_value_length("values", estimate.values, sample)
    old_values_full = estimate.old_values or estimate.values
    _validate_value_length("old_values", old_values_full, sample)

    rewards = _response_rewards(sample, estimate.rewards, response_positions)
    values = tuple(float(estimate.values[position]) for position in response_positions)
    old_values = tuple(float(old_values_full[position]) for position in response_positions)
    raw_advantages = _generalized_advantages(
        rewards,
        old_values,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    returns = tuple(
        advantage + value for advantage, value in zip(raw_advantages, old_values, strict=True)
    )
    return PPOValueTarget(
        task_id=sample.task_id,
        sample_index=sample.sample_index,
        response_positions=response_positions,
        rewards=rewards,
        values=values,
        old_values=old_values,
        raw_advantages=raw_advantages,
        advantages=raw_advantages,
        returns=returns,
    )


def _compute_sample_loss(
    sample: SequenceTrainingSample,
    target: PPOValueTarget,
    config: PPOLossConfig,
) -> PPOSampleLoss:
    if sample.logprobs is None:
        raise ValueError("PPO loss requires sample.logprobs")
    logprobs = sample.logprobs
    if logprobs.old_policy_logprobs is None:
        raise ValueError("PPO loss requires old_policy_logprobs")
    if config.kl_beta > 0 and logprobs.reference_logprobs is None:
        raise ValueError("PPO KL penalty requires reference_logprobs when kl_beta > 0")

    lower_clip = 1.0 - config.clip_epsilon
    upper_clip = 1.0 + config.clip_epsilon
    policy_terms: list[float] = []
    ratios: list[float] = []
    clipped_ratios: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []
    approx_kls: list[float] = []
    clipped_policy_tokens = 0
    clipped_value_tokens = 0

    for offset, index in enumerate(target.response_positions):
        advantage = target.advantages[offset]
        log_ratio = logprobs.policy_logprobs[index] - logprobs.old_policy_logprobs[index]
        ratio = _exp_clamped(log_ratio, config.max_log_ratio)
        clipped_ratio = min(max(ratio, lower_clip), upper_clip)
        policy_terms.append(min(ratio * advantage, clipped_ratio * advantage))
        ratios.append(ratio)
        clipped_ratios.append(clipped_ratio)
        if abs(ratio - clipped_ratio) > 1e-12:
            clipped_policy_tokens += 1

        value_loss, clipped_value = _value_loss_term(
            value=target.values[offset],
            old_value=target.old_values[offset],
            value_return=target.returns[offset],
            config=config,
        )
        value_losses.append(value_loss)
        if clipped_value:
            clipped_value_tokens += 1

        if logprobs.entropy is not None:
            entropies.append(logprobs.entropy[index])
        if logprobs.reference_logprobs is not None:
            reference_delta = logprobs.reference_logprobs[index] - logprobs.policy_logprobs[index]
            approx_kls.append(
                _exp_clamped(reference_delta, config.max_log_ratio) - reference_delta - 1.0
            )

    policy_loss = -fmean(policy_terms)
    value_loss = config.value_loss_coef * fmean(value_losses)
    mean_entropy = fmean(entropies) if entropies else 0.0
    entropy_loss = -config.entropy_coef * mean_entropy
    mean_approx_kl = fmean(approx_kls) if approx_kls else 0.0
    kl_loss = config.kl_beta * mean_approx_kl
    total_loss = policy_loss + value_loss + entropy_loss + kl_loss
    return PPOSampleLoss(
        task_id=sample.task_id,
        sample_index=sample.sample_index,
        response_tokens=target.response_tokens,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        kl_loss=kl_loss,
        total_loss=total_loss,
        mean_ratio=fmean(ratios),
        mean_clipped_ratio=fmean(clipped_ratios),
        mean_advantage=fmean(target.advantages),
        mean_return=fmean(target.returns),
        mean_entropy=mean_entropy,
        mean_approx_kl=mean_approx_kl,
        clipped_policy_tokens=clipped_policy_tokens,
        clipped_value_tokens=clipped_value_tokens,
    )


def _value_loss_term(
    *,
    value: float,
    old_value: float,
    value_return: float,
    config: PPOLossConfig,
) -> tuple[float, bool]:
    unclipped = (value - value_return) ** 2
    if config.value_clip_epsilon is None:
        return unclipped, False
    clipped_value = old_value + min(
        max(value - old_value, -config.value_clip_epsilon),
        config.value_clip_epsilon,
    )
    clipped = (clipped_value - value_return) ** 2
    return max(unclipped, clipped), clipped > unclipped + 1e-12


def _generalized_advantages(
    rewards: tuple[float, ...],
    values: tuple[float, ...],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[float, ...]:
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        next_advantage = delta + gamma * gae_lambda * next_advantage
        advantages[index] = next_advantage
    return tuple(advantages)


def _normalize_target_advantages(
    targets: tuple[PPOValueTarget, ...],
    *,
    min_std: float = 1e-8,
) -> tuple[PPOValueTarget, ...]:
    advantages = tuple(value for target in targets for value in target.raw_advantages)
    if not advantages:
        return targets
    mean_value = fmean(advantages)
    std = _population_std(advantages, mean_value)
    if std <= min_std:
        normalized = [0.0 for _ in advantages]
    else:
        normalized = [(value - mean_value) / std for value in advantages]

    output: list[PPOValueTarget] = []
    offset = 0
    for target in targets:
        count = target.response_tokens
        output.append(
            PPOValueTarget(
                task_id=target.task_id,
                sample_index=target.sample_index,
                response_positions=target.response_positions,
                rewards=target.rewards,
                values=target.values,
                old_values=target.old_values,
                raw_advantages=target.raw_advantages,
                advantages=tuple(normalized[offset : offset + count]),
                returns=target.returns,
            )
        )
        offset += count
    return tuple(output)


def _estimate_for(
    position: int,
    sample: SequenceTrainingSample,
    estimates: ValueEstimateInput,
) -> PPOValueEstimate:
    if isinstance(estimates, Mapping):
        key = (sample.task_id, sample.sample_index)
        if key not in estimates:
            raise ValueError(
                f"missing PPO value estimate for {sample.task_id}:{sample.sample_index}"
            )
        return estimates[key]
    if position >= len(estimates):
        raise ValueError("missing PPO value estimate for batch position")
    return estimates[position]


def _response_positions(sample: SequenceTrainingSample) -> tuple[int, ...]:
    return tuple(index for index, include in enumerate(sample.masks.response_mask) if include)


def _response_rewards(
    sample: SequenceTrainingSample,
    rewards: tuple[float, ...] | None,
    response_positions: tuple[int, ...],
) -> tuple[float, ...]:
    if rewards is None:
        sparse_rewards = [0.0] * len(response_positions)
        sparse_rewards[-1] = float(sample.reward)
        return tuple(sparse_rewards)
    if len(rewards) == len(sample.input_ids):
        return tuple(float(rewards[position]) for position in response_positions)
    if len(rewards) == len(response_positions):
        return tuple(float(reward) for reward in rewards)
    raise ValueError("rewards must match input_ids length or response token count")


def _validate_value_length(
    name: str,
    values: tuple[float, ...],
    sample: SequenceTrainingSample,
) -> None:
    if len(values) != len(sample.input_ids):
        raise ValueError(f"{name} length must match sample.input_ids")


def _exp_clamped(value: float, max_abs_value: float) -> float:
    return math.exp(min(max(value, -max_abs_value), max_abs_value))


def _population_std(values: tuple[float, ...], mean_value: float) -> float:
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)
