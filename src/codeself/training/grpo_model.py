"""Torch model-forward helpers for GRPO tensor batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from codeself.training.common import TrainingBatch
from codeself.training.grpo_torch import GRPOTensorBatch, require_torch


@dataclass(frozen=True)
class PaddedTrainingTensors:
    """Padded tensors derived from `TrainingBatch` records."""

    input_ids: Any
    attention_mask: Any
    response_mask: Any
    advantages: Any
    old_policy_logprobs: Any | None = None
    reference_logprobs: Any | None = None


def pad_training_batch_tensors(
    batch: TrainingBatch,
    *,
    device: str | None = None,
    dtype: Any | None = None,
    pad_token_id: int = 0,
) -> PaddedTrainingTensors:
    """Pad training samples into tensors for causal-LM forward passes."""

    torch = require_torch()
    if dtype is None:
        dtype = torch.float32
    max_tokens = max(len(sample.input_ids) for sample in batch.samples)
    input_rows: list[list[int]] = []
    attention_rows: list[list[float]] = []
    response_rows: list[list[float]] = []
    advantages: list[float] = []
    old_rows: list[list[float]] = []
    reference_rows: list[list[float]] = []
    old_presence: list[bool] = []
    reference_presence: list[bool] = []

    for sample in batch.samples:
        if sample.advantage is None:
            raise ValueError("GRPO model batch requires sample.advantage")
        pad_tokens = max_tokens - len(sample.input_ids)
        input_rows.append(
            [int(token_id) for token_id in sample.input_ids] + [pad_token_id] * pad_tokens
        )
        attention_rows.append([1.0] * len(sample.input_ids) + [0.0] * pad_tokens)
        response_rows.append(_padded_float_row(sample.masks.response_mask, pad_tokens))
        advantages.append(float(sample.advantage))
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
    return PaddedTrainingTensors(
        input_ids=torch.tensor(input_rows, dtype=torch.long, **tensor_kwargs),
        attention_mask=torch.tensor(attention_rows, **float_kwargs),
        response_mask=torch.tensor(response_rows, **float_kwargs),
        advantages=torch.tensor(advantages, **float_kwargs),
        old_policy_logprobs=old_policy_logprobs,
        reference_logprobs=reference_logprobs,
    )


def gather_causal_lm_token_logprobs(
    logits: Any,
    input_ids: Any,
    attention_mask: Any | None = None,
) -> Any:
    """Gather next-token logprobs aligned to each token position."""

    torch = require_torch()
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, tokens, vocab]")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must agree on batch and token dimensions")
    if input_ids.numel() and int(input_ids.max().detach().cpu().item()) >= logits.shape[-1]:
        raise ValueError("input_ids contain token ids outside the logits vocabulary")

    token_logprobs = torch.zeros(
        input_ids.shape,
        dtype=logits.dtype,
        device=logits.device,
    )
    if input_ids.shape[1] > 1:
        # Memory-efficient: gather the selected-token logit and subtract the
        # logsumexp over the vocab, instead of materializing a full
        # [batch, tokens, vocab] log_softmax tensor (which OOMs on real
        # vocabularies of 100k+ tokens).
        shift_logits = logits[:, :-1, :]
        next_token_ids = input_ids[:, 1:].unsqueeze(-1)
        selected = shift_logits.gather(-1, next_token_ids).squeeze(-1)
        logsumexp = torch.logsumexp(shift_logits, dim=-1)
        token_logprobs[:, 1:] = selected - logsumexp
    if attention_mask is not None:
        token_logprobs = token_logprobs * attention_mask.to(dtype=token_logprobs.dtype)
    return token_logprobs


def build_grpo_tensor_batch_from_model(
    batch: TrainingBatch,
    *,
    policy_model: Any,
    old_policy_model: Any | None = None,
    reference_model: Any | None = None,
    device: str | None = None,
    dtype: Any | None = None,
    pad_token_id: int = 0,
    microbatch_size: int = 0,
) -> GRPOTensorBatch:
    """Build a differentiable GRPO tensor batch from model forward passes.

    Forward passes are split into microbatches of ``microbatch_size`` samples
    (0 = whole batch) and each microbatch is padded to its own local maximum
    length, so peak activation memory scales with the microbatch rather than
    the full group. This keeps real-vocabulary models within MPS/GPU memory.
    """

    torch = require_torch()
    samples = list(batch.samples)
    global_max = max(len(sample.input_ids) for sample in samples)
    chunk = microbatch_size if microbatch_size and microbatch_size > 0 else len(samples)

    policy_chunks: list[Any] = []
    old_chunks: list[Any] = []
    reference_chunks: list[Any] = []
    mask_chunks: list[Any] = []
    advantage_chunks: list[Any] = []
    has_reference = False

    for start in range(0, len(samples), chunk):
        sub_batch = TrainingBatch(tuple(samples[start : start + chunk]))
        padded = pad_training_batch_tensors(
            sub_batch,
            device=device,
            dtype=dtype,
            pad_token_id=pad_token_id,
        )
        policy_logits = _model_logits(
            policy_model,
            input_ids=padded.input_ids,
            attention_mask=padded.attention_mask,
        )
        policy_logprobs = gather_causal_lm_token_logprobs(
            policy_logits,
            padded.input_ids,
            padded.attention_mask,
        )
        del policy_logits
        policy_chunks.append(_pad_token_dim(torch, policy_logprobs, global_max))

        if old_policy_model is not None:
            with torch.no_grad():
                old_logits = _model_logits(
                    old_policy_model,
                    input_ids=padded.input_ids,
                    attention_mask=padded.attention_mask,
                )
                old_logprobs = gather_causal_lm_token_logprobs(
                    old_logits,
                    padded.input_ids,
                    padded.attention_mask,
                )
                del old_logits
        elif padded.old_policy_logprobs is not None:
            old_logprobs = padded.old_policy_logprobs
        else:
            old_logprobs = policy_logprobs.detach()
        old_chunks.append(_pad_token_dim(torch, old_logprobs, global_max))

        chunk_reference = padded.reference_logprobs
        if reference_model is not None:
            with torch.no_grad():
                reference_logits = _model_logits(
                    reference_model,
                    input_ids=padded.input_ids,
                    attention_mask=padded.attention_mask,
                )
                chunk_reference = gather_causal_lm_token_logprobs(
                    reference_logits,
                    padded.input_ids,
                    padded.attention_mask,
                )
                del reference_logits
        if chunk_reference is not None:
            has_reference = True
            reference_chunks.append(_pad_token_dim(torch, chunk_reference, global_max))

        mask_chunks.append(_pad_token_dim(torch, padded.response_mask, global_max))
        advantage_chunks.append(padded.advantages)

    policy_logprobs = torch.cat(policy_chunks, dim=0)
    old_policy_logprobs = torch.cat(old_chunks, dim=0).detach()
    response_mask = torch.cat(mask_chunks, dim=0)
    advantages = torch.cat(advantage_chunks, dim=0)
    reference_logprobs = (
        torch.cat(reference_chunks, dim=0).detach() if has_reference else None
    )

    return GRPOTensorBatch(
        policy_logprobs=policy_logprobs,
        old_policy_logprobs=old_policy_logprobs,
        response_mask=response_mask,
        advantages=advantages,
        reference_logprobs=reference_logprobs,
    )


def _pad_token_dim(torch: Any, tensor: Any, target_tokens: int) -> Any:
    """Right-pad the token dimension of a `[batch, tokens]` tensor with zeros."""

    current = tensor.shape[1]
    if current >= target_tokens:
        return tensor
    return torch.nn.functional.pad(tensor, (0, target_tokens - current))


def _model_logits(model: Any, *, input_ids: Any, attention_mask: Any) -> Any:
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if isinstance(output, tuple):
        return output[0]
    return output


def _padded_float_row(values: tuple[float, ...] | tuple[int, ...], pad_tokens: int) -> list[float]:
    return [float(value) for value in values] + [0.0] * pad_tokens
