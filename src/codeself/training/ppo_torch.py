"""Optional Torch implementation of the PPO objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeself.training.common import TrainingBatch
from codeself.training.grpo_torch import require_torch
from codeself.training.ppo_loss import (
    PPOLossConfig,
    ValueEstimateInput,
    prepare_ppo_value_targets,
)


@dataclass(frozen=True)
class PPOTensorBatch:
    """Padded tensor inputs for the PPO objective."""

    policy_logprobs: Any
    old_policy_logprobs: Any
    response_mask: Any
    advantages: Any
    returns: Any
    values: Any
    old_values: Any
    entropy: Any | None = None
    reference_logprobs: Any | None = None


@dataclass(frozen=True)
class PPOTensorLossResult:
    """Differentiable PPO loss plus detached scalar diagnostics."""

    loss: Any
    policy_loss: Any
    value_loss: Any
    entropy_loss: Any
    kl_loss: Any
    total_response_tokens: int
    mean_ratio: float
    mean_clipped_ratio: float
    mean_advantage: float
    mean_return: float
    mean_entropy: float
    mean_approx_kl: float
    clipped_policy_fraction: float
    clipped_value_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": float(self.loss.detach().cpu().item()),
            "policy_loss": float(self.policy_loss.detach().cpu().item()),
            "value_loss": float(self.value_loss.detach().cpu().item()),
            "entropy_loss": float(self.entropy_loss.detach().cpu().item()),
            "kl_loss": float(self.kl_loss.detach().cpu().item()),
            "total_response_tokens": self.total_response_tokens,
            "mean_ratio": self.mean_ratio,
            "mean_clipped_ratio": self.mean_clipped_ratio,
            "mean_advantage": self.mean_advantage,
            "mean_return": self.mean_return,
            "mean_entropy": self.mean_entropy,
            "mean_approx_kl": self.mean_approx_kl,
            "clipped_policy_fraction": self.clipped_policy_fraction,
            "clipped_value_fraction": self.clipped_value_fraction,
        }


def build_ppo_tensor_batch(
    batch: TrainingBatch,
    value_estimates: ValueEstimateInput,
    *,
    config: PPOLossConfig | None = None,
    device: str | None = None,
    dtype: Any | None = None,
) -> PPOTensorBatch:
    """Convert a `TrainingBatch` and PPO value estimates into padded tensors."""

    torch = require_torch()
    loss_config = config or PPOLossConfig()
    if dtype is None:
        dtype = torch.float32
    value_targets = prepare_ppo_value_targets(batch, value_estimates, config=loss_config)
    max_tokens = max(len(sample.input_ids) for sample in batch.samples)
    policy_rows: list[list[float]] = []
    old_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    entropy_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    advantage_rows: list[list[float]] = []
    return_rows: list[list[float]] = []
    value_rows: list[list[float]] = []
    old_value_rows: list[list[float]] = []
    reference_presence: list[bool] = []

    for sample, target in zip(batch.samples, value_targets.targets, strict=True):
        if sample.logprobs is None:
            raise ValueError("PPO tensor batch requires sample.logprobs")
        if sample.logprobs.old_policy_logprobs is None:
            raise ValueError("PPO tensor batch requires old_policy_logprobs")
        pad_tokens = max_tokens - len(sample.input_ids)
        policy_rows.append(_padded_float_row(sample.logprobs.policy_logprobs, pad_tokens))
        old_rows.append(_padded_float_row(sample.logprobs.old_policy_logprobs, pad_tokens))
        entropy_rows.append(
            _padded_float_row(
                sample.logprobs.entropy or tuple(0.0 for _ in sample.input_ids),
                pad_tokens,
            )
        )
        has_reference = sample.logprobs.reference_logprobs is not None
        reference_presence.append(has_reference)
        if has_reference:
            reference_rows.append(
                _padded_float_row(sample.logprobs.reference_logprobs or (), pad_tokens)
            )
        mask_rows.append(_padded_float_row(sample.masks.response_mask, pad_tokens))
        advantage_rows.append(_target_row(target.response_positions, target.advantages, max_tokens))
        return_rows.append(_target_row(target.response_positions, target.returns, max_tokens))
        value_rows.append(_target_row(target.response_positions, target.values, max_tokens))
        old_value_rows.append(_target_row(target.response_positions, target.old_values, max_tokens))

    if any(reference_presence) and not all(reference_presence):
        raise ValueError("reference_logprobs must be present for every sample or none")

    tensor_kwargs: dict[str, Any] = {"dtype": dtype}
    if device is not None:
        tensor_kwargs["device"] = device
    reference_logprobs = (
        torch.tensor(reference_rows, **tensor_kwargs) if all(reference_presence) else None
    )
    return PPOTensorBatch(
        policy_logprobs=torch.tensor(policy_rows, **tensor_kwargs),
        old_policy_logprobs=torch.tensor(old_rows, **tensor_kwargs),
        response_mask=torch.tensor(mask_rows, **tensor_kwargs),
        advantages=torch.tensor(advantage_rows, **tensor_kwargs),
        returns=torch.tensor(return_rows, **tensor_kwargs),
        values=torch.tensor(value_rows, **tensor_kwargs),
        old_values=torch.tensor(old_value_rows, **tensor_kwargs),
        entropy=torch.tensor(entropy_rows, **tensor_kwargs),
        reference_logprobs=reference_logprobs,
    )


def compute_ppo_tensor_loss(
    batch: PPOTensorBatch,
    *,
    config: PPOLossConfig | None = None,
) -> PPOTensorLossResult:
    """Compute a differentiable clipped PPO objective with Torch tensors."""

    torch = require_torch()
    loss_config = config or PPOLossConfig()
    _validate_tensor_batch(torch, batch)

    policy_logprobs = batch.policy_logprobs
    old_policy_logprobs = batch.old_policy_logprobs
    response_mask = batch.response_mask.to(dtype=policy_logprobs.dtype)
    advantages = batch.advantages.to(dtype=policy_logprobs.dtype)
    returns = batch.returns.to(dtype=policy_logprobs.dtype)
    values = batch.values.to(dtype=policy_logprobs.dtype)
    old_values = batch.old_values.to(dtype=policy_logprobs.dtype)
    response_counts = response_mask.sum(dim=1)
    if bool(torch.any(response_counts <= 0).detach().cpu().item()):
        raise ValueError("PPO tensor loss requires at least one response token per sample")
    token_count = response_counts.sum()

    log_ratio = torch.clamp(
        policy_logprobs - old_policy_logprobs,
        min=-loss_config.max_log_ratio,
        max=loss_config.max_log_ratio,
    )
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - loss_config.clip_epsilon,
        max=1.0 + loss_config.clip_epsilon,
    )
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -_batch_mean_of_sequence_means(surrogate, response_mask, response_counts)

    value_errors = (values - returns) ** 2
    if loss_config.value_clip_epsilon is None:
        value_loss_terms = value_errors
        clipped_value_mask = torch.zeros_like(response_mask)
    else:
        value_delta = torch.clamp(
            values - old_values,
            min=-loss_config.value_clip_epsilon,
            max=loss_config.value_clip_epsilon,
        )
        clipped_values = old_values + value_delta
        clipped_value_errors = (clipped_values - returns) ** 2
        value_loss_terms = torch.maximum(value_errors, clipped_value_errors)
        clipped_value_mask = (clipped_value_errors > value_errors + 1e-12).to(
            response_mask.dtype
        )
    value_loss = loss_config.value_loss_coef * _batch_mean_of_sequence_means(
        value_loss_terms,
        response_mask,
        response_counts,
    )

    entropy = (
        batch.entropy.to(dtype=policy_logprobs.dtype)
        if batch.entropy is not None
        else torch.zeros_like(policy_logprobs)
    )
    mean_entropy_tensor = _batch_mean_of_sequence_means(entropy, response_mask, response_counts)
    entropy_loss = -loss_config.entropy_coef * mean_entropy_tensor

    if batch.reference_logprobs is None:
        if loss_config.kl_beta > 0:
            raise ValueError("PPO tensor KL penalty requires reference_logprobs")
        approx_kl = torch.zeros_like(policy_logprobs)
    else:
        reference_delta = torch.clamp(
            batch.reference_logprobs - policy_logprobs,
            min=-loss_config.max_log_ratio,
            max=loss_config.max_log_ratio,
        )
        approx_kl = torch.exp(reference_delta) - reference_delta - 1.0
    mean_approx_kl_tensor = _batch_mean_of_sequence_means(
        approx_kl,
        response_mask,
        response_counts,
    )
    kl_loss = loss_config.kl_beta * mean_approx_kl_tensor
    loss = policy_loss + value_loss + entropy_loss + kl_loss

    clipped_policy_tokens = (
        (torch.abs(clipped_ratio - ratio) > 1e-12).to(response_mask.dtype) * response_mask
    ).sum()
    clipped_value_tokens = (clipped_value_mask * response_mask).sum()
    return PPOTensorLossResult(
        loss=loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_loss=entropy_loss,
        kl_loss=kl_loss,
        total_response_tokens=int(token_count.detach().cpu().item()),
        mean_ratio=_detach_float(
            _batch_mean_of_sequence_means(ratio, response_mask, response_counts)
        ),
        mean_clipped_ratio=_detach_float(
            _batch_mean_of_sequence_means(clipped_ratio, response_mask, response_counts)
        ),
        mean_advantage=_detach_float(
            _batch_mean_of_sequence_means(advantages, response_mask, response_counts)
        ),
        mean_return=_detach_float(
            _batch_mean_of_sequence_means(returns, response_mask, response_counts)
        ),
        mean_entropy=_detach_float(mean_entropy_tensor),
        mean_approx_kl=_detach_float(mean_approx_kl_tensor),
        clipped_policy_fraction=_detach_float(
            clipped_policy_tokens / token_count.clamp_min(1.0)
        ),
        clipped_value_fraction=_detach_float(
            clipped_value_tokens / token_count.clamp_min(1.0)
        ),
    )


def _validate_tensor_batch(torch: Any, batch: PPOTensorBatch) -> None:
    tensor_fields = (
        "policy_logprobs",
        "old_policy_logprobs",
        "response_mask",
        "advantages",
        "returns",
        "values",
        "old_values",
    )
    for field in tensor_fields:
        if not torch.is_tensor(getattr(batch, field)):
            raise TypeError(f"{field} must be a Torch tensor")
    if batch.policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must have shape [batch, tokens]")
    for field in tensor_fields[1:]:
        if getattr(batch, field).shape != batch.policy_logprobs.shape:
            raise ValueError(f"{field} shape must match policy_logprobs")
    if batch.entropy is not None:
        if not torch.is_tensor(batch.entropy):
            raise TypeError("entropy must be a Torch tensor")
        if batch.entropy.shape != batch.policy_logprobs.shape:
            raise ValueError("entropy shape must match policy_logprobs")
    if batch.reference_logprobs is not None:
        if not torch.is_tensor(batch.reference_logprobs):
            raise TypeError("reference_logprobs must be a Torch tensor")
        if batch.reference_logprobs.shape != batch.policy_logprobs.shape:
            raise ValueError("reference_logprobs shape must match policy_logprobs")


def _target_row(
    positions: tuple[int, ...],
    values: tuple[float, ...],
    max_tokens: int,
) -> list[float]:
    row = [0.0] * max_tokens
    for position, value in zip(positions, values, strict=True):
        row[position] = float(value)
    return row


def _batch_mean_of_sequence_means(values: Any, mask: Any, response_counts: Any) -> Any:
    sequence_means = (values * mask).sum(dim=1) / response_counts
    return sequence_means.mean()


def _detach_float(value: Any) -> float:
    return float(value.detach().cpu().item())


def _padded_float_row(values: tuple[float, ...] | tuple[int, ...], pad_tokens: int) -> list[float]:
    return [float(value) for value in values] + [0.0] * pad_tokens
