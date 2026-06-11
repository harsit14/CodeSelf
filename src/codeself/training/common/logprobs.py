"""Token logprob records and response-level summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from codeself.training.common.masking import TokenMask, mask_mean


@dataclass(frozen=True)
class TokenLogprobs:
    """Policy, old-policy, and reference logprobs for one sequence."""

    token_ids: tuple[int, ...]
    policy_logprobs: tuple[float, ...]
    reference_logprobs: tuple[float, ...] | None = None
    old_policy_logprobs: tuple[float, ...] | None = None
    entropy: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        expected = len(self.token_ids)
        for name, values in (
            ("policy_logprobs", self.policy_logprobs),
            ("reference_logprobs", self.reference_logprobs),
            ("old_policy_logprobs", self.old_policy_logprobs),
            ("entropy", self.entropy),
        ):
            if values is not None and len(values) != expected:
                raise ValueError(f"{name} length must match token_ids")

    def to_dict(self) -> dict[str, object]:
        return {
            "token_ids": list(self.token_ids),
            "policy_logprobs": list(self.policy_logprobs),
            "reference_logprobs": list(self.reference_logprobs) if self.reference_logprobs else None,
            "old_policy_logprobs": list(self.old_policy_logprobs) if self.old_policy_logprobs else None,
            "entropy": list(self.entropy) if self.entropy else None,
        }


@dataclass(frozen=True)
class LogprobSummary:
    """Response-token aggregate logprob diagnostics."""

    response_tokens: int
    mean_policy_logprob: float
    mean_reference_logprob: float | None
    mean_old_policy_logprob: float | None
    mean_kl_to_reference: float | None
    mean_importance_ratio: float | None
    mean_entropy: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "response_tokens": self.response_tokens,
            "mean_policy_logprob": self.mean_policy_logprob,
            "mean_reference_logprob": self.mean_reference_logprob,
            "mean_old_policy_logprob": self.mean_old_policy_logprob,
            "mean_kl_to_reference": self.mean_kl_to_reference,
            "mean_importance_ratio": self.mean_importance_ratio,
            "mean_entropy": self.mean_entropy,
        }


def summarize_logprobs(logprobs: TokenLogprobs, mask: TokenMask) -> LogprobSummary:
    """Summarize logprob diagnostics over response tokens only."""

    if len(logprobs.token_ids) != mask.token_count:
        raise ValueError("logprob sequence length must match mask length")
    response_mask = mask.response_mask
    mean_policy = mask_mean(logprobs.policy_logprobs, response_mask)
    mean_reference = None
    mean_kl = None
    if logprobs.reference_logprobs is not None:
        mean_reference = mask_mean(logprobs.reference_logprobs, response_mask)
        kl_values = tuple(
            policy - reference
            for policy, reference in zip(
                logprobs.policy_logprobs,
                logprobs.reference_logprobs,
                strict=True,
            )
        )
        mean_kl = mask_mean(kl_values, response_mask)
    mean_old = None
    mean_ratio = None
    if logprobs.old_policy_logprobs is not None:
        mean_old = mask_mean(logprobs.old_policy_logprobs, response_mask)
        ratio_values = tuple(
            exp(min(20.0, max(-20.0, policy - old)))
            for policy, old in zip(
                logprobs.policy_logprobs,
                logprobs.old_policy_logprobs,
                strict=True,
            )
        )
        mean_ratio = mask_mean(ratio_values, response_mask)
    mean_entropy = mask_mean(logprobs.entropy, response_mask) if logprobs.entropy else None
    return LogprobSummary(
        response_tokens=mask.response_tokens,
        mean_policy_logprob=mean_policy,
        mean_reference_logprob=mean_reference,
        mean_old_policy_logprob=mean_old,
        mean_kl_to_reference=mean_kl,
        mean_importance_ratio=mean_ratio,
        mean_entropy=mean_entropy,
    )
