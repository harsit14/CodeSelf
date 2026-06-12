"""Torch model-forward helpers for PPO tensor batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from codeself.training.common import TrainingBatch
from codeself.training.grpo_model import gather_causal_lm_token_logprobs
from codeself.training.grpo_torch import require_torch
from codeself.training.ppo_loss import (
    PPOLossConfig,
    PPOValueEstimate,
    ValueEstimateInput,
    prepare_ppo_value_targets,
)
from codeself.training.ppo_torch import PPOTensorBatch


@dataclass(frozen=True)
class PaddedPPOTrainingTensors:
    """Padded tensors derived from PPO `TrainingBatch` records."""

    input_ids: Any
    attention_mask: Any
    response_mask: Any
    old_policy_logprobs: Any | None = None
    reference_logprobs: Any | None = None


def pad_ppo_training_batch_tensors(
    batch: TrainingBatch,
    *,
    device: str | None = None,
    dtype: Any | None = None,
    pad_token_id: int = 0,
) -> PaddedPPOTrainingTensors:
    """Pad PPO training samples without requiring precomputed advantages."""

    torch = require_torch()
    if dtype is None:
        dtype = torch.float32
    max_tokens = max(len(sample.input_ids) for sample in batch.samples)
    input_rows: list[list[int]] = []
    attention_rows: list[list[float]] = []
    response_rows: list[list[float]] = []
    old_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    old_presence: list[bool] = []
    reference_presence: list[bool] = []

    for sample in batch.samples:
        pad_tokens = max_tokens - len(sample.input_ids)
        input_rows.append(
            [int(token_id) for token_id in sample.input_ids] + [pad_token_id] * pad_tokens
        )
        attention_rows.append([1.0] * len(sample.input_ids) + [0.0] * pad_tokens)
        response_rows.append(_padded_float_row(sample.masks.response_mask, pad_tokens))
        has_old = sample.logprobs is not None and sample.logprobs.old_policy_logprobs is not None
        has_reference = (
            sample.logprobs is not None
            and sample.logprobs.reference_logprobs is not None
        )
        old_presence.append(has_old)
        reference_presence.append(has_reference)
        if has_old:
            old_rows.append(
                _padded_float_row(sample.logprobs.old_policy_logprobs or (), pad_tokens)
            )
        if has_reference:
            reference_rows.append(
                _padded_float_row(sample.logprobs.reference_logprobs or (), pad_tokens)
            )

    if any(old_presence) and not all(old_presence):
        raise ValueError("old_policy_logprobs must be present for every sample or none")
    if any(reference_presence) and not all(reference_presence):
        raise ValueError("reference_logprobs must be present for every sample or none")

    tensor_kwargs: dict[str, Any] = {"device": device} if device is not None else {}
    float_kwargs = {**tensor_kwargs, "dtype": dtype}
    old_policy_logprobs = torch.tensor(old_rows, **float_kwargs) if all(old_presence) else None
    reference_logprobs = (
        torch.tensor(reference_rows, **float_kwargs) if all(reference_presence) else None
    )
    return PaddedPPOTrainingTensors(
        input_ids=torch.tensor(input_rows, dtype=torch.long, **tensor_kwargs),
        attention_mask=torch.tensor(attention_rows, **float_kwargs),
        response_mask=torch.tensor(response_rows, **float_kwargs),
        old_policy_logprobs=old_policy_logprobs,
        reference_logprobs=reference_logprobs,
    )


def gather_causal_lm_token_entropy(
    logits: Any,
    attention_mask: Any | None = None,
) -> Any:
    """Compute next-token categorical entropy aligned to token positions."""

    torch = require_torch()
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    token_entropy = torch.zeros(
        logits.shape[:2],
        dtype=logits.dtype,
        device=logits.device,
    )
    if logits.shape[1] > 1:
        # Memory-efficient entropy: H = logsumexp(l) - sum(softmax(l) * l).
        # This avoids holding both a full-vocab log_softmax and its exp at once
        # (which doubles the [batch, tokens, vocab] footprint and OOMs on real
        # vocabularies).
        shift_logits = logits[:, :-1, :]
        logsumexp = torch.logsumexp(shift_logits, dim=-1)
        softmax = torch.softmax(shift_logits, dim=-1)
        weighted = (softmax * shift_logits).sum(dim=-1)
        token_entropy[:, 1:] = logsumexp - weighted
    if attention_mask is not None:
        token_entropy = token_entropy * attention_mask.to(dtype=token_entropy.dtype)
    return token_entropy


def build_ppo_tensor_batch_from_model(
    batch: TrainingBatch,
    *,
    policy_model: Any,
    value_model: Any | None = None,
    old_policy_model: Any | None = None,
    old_value_model: Any | None = None,
    reference_model: Any | None = None,
    value_estimates: ValueEstimateInput | None = None,
    config: PPOLossConfig | None = None,
    device: str | None = None,
    dtype: Any | None = None,
    pad_token_id: int = 0,
) -> PPOTensorBatch:
    """Build a differentiable PPO tensor batch from model forward passes."""

    torch = require_torch()
    loss_config = config or PPOLossConfig()
    if dtype is None:
        dtype = torch.float32
    padded = pad_ppo_training_batch_tensors(
        batch,
        device=device,
        dtype=dtype,
        pad_token_id=pad_token_id,
    )
    policy_output = _model_output(
        policy_model,
        input_ids=padded.input_ids,
        attention_mask=padded.attention_mask,
    )
    policy_logits = _output_logits(policy_output)
    policy_logprobs = gather_causal_lm_token_logprobs(
        policy_logits,
        padded.input_ids,
        padded.attention_mask,
    )
    values = _model_values_from_output_or_model(
        policy_output=policy_output,
        value_model=value_model,
        input_ids=padded.input_ids,
        attention_mask=padded.attention_mask,
        dtype=dtype,
    )

    if old_policy_model is not None:
        with torch.no_grad():
            old_policy_logits = _output_logits(
                _model_output(
                    old_policy_model,
                    input_ids=padded.input_ids,
                    attention_mask=padded.attention_mask,
                )
            )
            old_policy_logprobs = gather_causal_lm_token_logprobs(
                old_policy_logits,
                padded.input_ids,
                padded.attention_mask,
            )
    elif padded.old_policy_logprobs is not None:
        old_policy_logprobs = padded.old_policy_logprobs
    else:
        old_policy_logprobs = policy_logprobs.detach()

    if old_value_model is not None:
        with torch.no_grad():
            old_values = _model_values_from_output_or_model(
                policy_output=None,
                value_model=old_value_model,
                input_ids=padded.input_ids,
                attention_mask=padded.attention_mask,
                dtype=dtype,
            )
    else:
        old_values = values.detach()

    reference_logprobs = padded.reference_logprobs
    if reference_model is not None:
        with torch.no_grad():
            reference_logits = _output_logits(
                _model_output(
                    reference_model,
                    input_ids=padded.input_ids,
                    attention_mask=padded.attention_mask,
                )
            )
            reference_logprobs = gather_causal_lm_token_logprobs(
                reference_logits,
                padded.input_ids,
                padded.attention_mask,
            )

    targets = prepare_ppo_value_targets(
        batch,
        value_estimates
        or _value_estimates_from_old_values(batch, old_values.detach()),
        config=loss_config,
    )
    max_tokens = padded.input_ids.shape[1]
    tensor_kwargs = {"dtype": dtype, "device": padded.input_ids.device}
    advantage_rows = [
        _target_row(target.response_positions, target.advantages, max_tokens)
        for target in targets.targets
    ]
    return_rows = [
        _target_row(target.response_positions, target.returns, max_tokens)
        for target in targets.targets
    ]
    old_value_rows = [
        _target_row(target.response_positions, target.old_values, max_tokens)
        for target in targets.targets
    ]

    return PPOTensorBatch(
        policy_logprobs=policy_logprobs,
        old_policy_logprobs=old_policy_logprobs.detach(),
        response_mask=padded.response_mask,
        advantages=torch.tensor(advantage_rows, **tensor_kwargs),
        returns=torch.tensor(return_rows, **tensor_kwargs),
        values=values,
        old_values=torch.tensor(old_value_rows, **tensor_kwargs),
        entropy=gather_causal_lm_token_entropy(policy_logits, padded.attention_mask),
        reference_logprobs=reference_logprobs.detach() if reference_logprobs is not None else None,
    )


def _model_output(model: Any, *, input_ids: Any, attention_mask: Any) -> Any:
    return model(input_ids=input_ids, attention_mask=attention_mask)


def _output_logits(output: Any) -> Any:
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output


def _model_values_from_output_or_model(
    *,
    policy_output: Any | None,
    value_model: Any | None,
    input_ids: Any,
    attention_mask: Any,
    dtype: Any,
) -> Any:
    if value_model is not None:
        values = _output_values(
            _model_output(value_model, input_ids=input_ids, attention_mask=attention_mask)
        )
    else:
        values = _output_values(policy_output)
    values = _normalize_value_tensor(values)
    if values.shape != input_ids.shape:
        raise ValueError("PPO value outputs must have shape [batch, tokens]")
    return values.to(dtype=dtype) * attention_mask.to(dtype=dtype)


def _output_values(output: Any) -> Any:
    if output is None:
        raise ValueError("PPO model batch requires a value model or policy output values")
    if _looks_like_tensor(output):
        return output
    for name in ("values", "value", "value_logits"):
        if isinstance(output, Mapping) and name in output:
            return output[name]
        if hasattr(output, name):
            value = getattr(output, name)
            if not callable(value):
                return value
    if isinstance(output, tuple) and len(output) > 1:
        return output[1]
    raise ValueError("PPO model batch requires a value model or policy output values")


def _normalize_value_tensor(values: Any) -> Any:
    if not _looks_like_tensor(values):
        raise TypeError("PPO value outputs must be Torch tensors")
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values.squeeze(-1)
    if values.ndim != 2:
        raise ValueError("PPO value outputs must have shape [batch, tokens]")
    return values


def _value_estimates_from_old_values(
    batch: TrainingBatch,
    old_values: Any,
) -> Sequence[PPOValueEstimate]:
    rows = old_values.detach().cpu().tolist()
    estimates: list[PPOValueEstimate] = []
    for sample, row in zip(batch.samples, rows, strict=True):
        values = tuple(float(value) for value in row[: len(sample.input_ids)])
        estimates.append(PPOValueEstimate(values=values, old_values=values))
    return tuple(estimates)


def _target_row(
    positions: tuple[int, ...],
    values: tuple[float, ...],
    max_tokens: int,
) -> list[float]:
    row = [0.0] * max_tokens
    for position, value in zip(positions, values, strict=True):
        row[position] = float(value)
    return row


def _padded_float_row(values: tuple[float, ...] | tuple[int, ...], pad_tokens: int) -> list[float]:
    return [float(value) for value in values] + [0.0] * pad_tokens


def _looks_like_tensor(value: Any) -> bool:
    return hasattr(value, "ndim") and hasattr(value, "shape") and hasattr(value, "to")
