"""Optional Torch implementation of the GRPO objective."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from codeself.training.common import TrainingBatch
from codeself.training.grpo_loss import GRPOLossConfig


@dataclass(frozen=True)
class GRPOTensorBatch:
    """Padded tensor inputs for the GRPO objective.

    Tensor fields are typed as `Any` so this module remains importable when
    Torch is not installed. Runtime validation happens inside the compute path.
    """

    policy_logprobs: Any
    old_policy_logprobs: Any
    response_mask: Any
    advantages: Any
    reference_logprobs: Any | None = None


@dataclass(frozen=True)
class GRPOTensorLossResult:
    """Differentiable GRPO loss plus detached scalar diagnostics."""

    loss: Any
    policy_loss: Any
    kl_loss: Any
    total_response_tokens: int
    mean_ratio: float
    mean_clipped_ratio: float
    mean_approx_kl: float
    clipped_token_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": float(self.loss.detach().cpu().item()),
            "policy_loss": float(self.policy_loss.detach().cpu().item()),
            "kl_loss": float(self.kl_loss.detach().cpu().item()),
            "total_response_tokens": self.total_response_tokens,
            "mean_ratio": self.mean_ratio,
            "mean_clipped_ratio": self.mean_clipped_ratio,
            "mean_approx_kl": self.mean_approx_kl,
            "clipped_token_fraction": self.clipped_token_fraction,
        }


def torch_training_available() -> bool:
    """Return whether Torch can be imported without importing it now."""

    return find_spec("torch") is not None


def require_torch() -> Any:
    """Import Torch or raise a clean optional-dependency error."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch GRPO requires the optional training dependency `torch`; "
            "install CodeSelf with the training extra before running tensor losses."
        ) from exc
    return torch


def build_grpo_tensor_batch(
    batch: TrainingBatch,
    *,
    device: str | None = None,
    dtype: Any | None = None,
) -> GRPOTensorBatch:
    """Convert a `TrainingBatch` into padded Torch tensors."""

    torch = require_torch()
    if dtype is None:
        dtype = torch.float32
    max_tokens = max(len(sample.input_ids) for sample in batch.samples)
    policy_rows: list[list[float]] = []
    old_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    advantages: list[float] = []
    reference_presence: list[bool] = []

    for sample in batch.samples:
        if sample.logprobs is None:
            raise ValueError("GRPO tensor batch requires sample.logprobs")
        if sample.logprobs.old_policy_logprobs is None:
            raise ValueError("GRPO tensor batch requires old_policy_logprobs")
        if sample.advantage is None:
            raise ValueError("GRPO tensor batch requires sample.advantage")
        pad_tokens = max_tokens - len(sample.input_ids)
        policy_rows.append(_padded_float_row(sample.logprobs.policy_logprobs, pad_tokens))
        old_rows.append(_padded_float_row(sample.logprobs.old_policy_logprobs, pad_tokens))
        mask_rows.append(_padded_float_row(sample.masks.response_mask, pad_tokens))
        advantages.append(float(sample.advantage))
        has_reference = sample.logprobs.reference_logprobs is not None
        reference_presence.append(has_reference)
        if has_reference:
            reference_rows.append(
                _padded_float_row(sample.logprobs.reference_logprobs or (), pad_tokens)
            )

    if any(reference_presence) and not all(reference_presence):
        raise ValueError("reference_logprobs must be present for every sample or none")

    tensor_kwargs: dict[str, Any] = {"dtype": dtype}
    if device is not None:
        tensor_kwargs["device"] = device
    reference_logprobs = (
        torch.tensor(reference_rows, **tensor_kwargs) if all(reference_presence) else None
    )
    return GRPOTensorBatch(
        policy_logprobs=torch.tensor(policy_rows, **tensor_kwargs),
        old_policy_logprobs=torch.tensor(old_rows, **tensor_kwargs),
        response_mask=torch.tensor(mask_rows, **tensor_kwargs),
        advantages=torch.tensor(advantages, **tensor_kwargs),
        reference_logprobs=reference_logprobs,
    )


def compute_grpo_tensor_loss(
    batch: GRPOTensorBatch,
    *,
    config: GRPOLossConfig | None = None,
) -> GRPOTensorLossResult:
    """Compute a differentiable clipped GRPO objective with Torch tensors."""

    torch = require_torch()
    loss_config = config or GRPOLossConfig()
    _validate_tensor_batch(torch, batch)

    policy_logprobs = batch.policy_logprobs
    old_policy_logprobs = batch.old_policy_logprobs
    response_mask = batch.response_mask.to(dtype=policy_logprobs.dtype)
    advantages = batch.advantages.to(dtype=policy_logprobs.dtype)
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(-1)
    if advantages.shape != (policy_logprobs.shape[0], 1):
        raise ValueError("advantages must have shape [batch] or [batch, 1]")

    response_counts = response_mask.sum(dim=1)
    if bool(torch.any(response_counts <= 0).detach().cpu().item()):
        raise ValueError("GRPO tensor loss requires at least one response token per sample")
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

    if batch.reference_logprobs is None:
        if loss_config.kl_beta > 0:
            raise ValueError("GRPO tensor KL penalty requires reference_logprobs")
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
    loss = policy_loss + kl_loss

    clipped_tokens = (
        (torch.abs(clipped_ratio - ratio) > 1e-12).to(response_mask.dtype) * response_mask
    ).sum()
    return GRPOTensorLossResult(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        total_response_tokens=int(token_count.detach().cpu().item()),
        mean_ratio=_detach_float(
            _batch_mean_of_sequence_means(ratio, response_mask, response_counts)
        ),
        mean_clipped_ratio=_detach_float(
            _batch_mean_of_sequence_means(clipped_ratio, response_mask, response_counts)
        ),
        mean_approx_kl=_detach_float(mean_approx_kl_tensor),
        clipped_token_fraction=_detach_float(clipped_tokens / token_count.clamp_min(1.0)),
    )


def _validate_tensor_batch(torch: Any, batch: GRPOTensorBatch) -> None:
    if not torch.is_tensor(batch.policy_logprobs):
        raise TypeError("policy_logprobs must be a Torch tensor")
    if not torch.is_tensor(batch.old_policy_logprobs):
        raise TypeError("old_policy_logprobs must be a Torch tensor")
    if not torch.is_tensor(batch.response_mask):
        raise TypeError("response_mask must be a Torch tensor")
    if not torch.is_tensor(batch.advantages):
        raise TypeError("advantages must be a Torch tensor")
    if batch.policy_logprobs.ndim != 2:
        raise ValueError("policy_logprobs must have shape [batch, tokens]")
    if batch.old_policy_logprobs.shape != batch.policy_logprobs.shape:
        raise ValueError("old_policy_logprobs shape must match policy_logprobs")
    if batch.response_mask.shape != batch.policy_logprobs.shape:
        raise ValueError("response_mask shape must match policy_logprobs")
    if batch.reference_logprobs is not None:
        if not torch.is_tensor(batch.reference_logprobs):
            raise TypeError("reference_logprobs must be a Torch tensor")
        if batch.reference_logprobs.shape != batch.policy_logprobs.shape:
            raise ValueError("reference_logprobs shape must match policy_logprobs")


def _batch_mean_of_sequence_means(values: Any, mask: Any, response_counts: Any) -> Any:
    sequence_means = (values * mask).sum(dim=1) / response_counts
    return sequence_means.mean()


def _detach_float(value: Any) -> float:
    return float(value.detach().cpu().item())


def _padded_float_row(values: tuple[float, ...] | tuple[int, ...], pad_tokens: int) -> list[float]:
    return [float(value) for value in values] + [0.0] * pad_tokens
