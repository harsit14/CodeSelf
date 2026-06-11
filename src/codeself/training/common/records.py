"""Training sequence records shared by GRPO and PPO implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

from codeself.training.common.logprobs import LogprobSummary, TokenLogprobs, summarize_logprobs
from codeself.training.common.masking import TokenMask


@dataclass(frozen=True)
class SequenceTrainingSample:
    """One tokenized prompt/response sample prepared for policy-gradient loss."""

    task_id: str
    sample_index: int
    input_ids: tuple[int, ...]
    masks: TokenMask
    reward: float
    logprobs: TokenLogprobs | None = None
    advantage: float | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.input_ids) != self.masks.token_count:
            raise ValueError("input_ids length must match masks")
        if self.logprobs is not None and len(self.logprobs.token_ids) != len(self.input_ids):
            raise ValueError("logprobs length must match input_ids")

    @property
    def response_tokens(self) -> int:
        return self.masks.response_tokens

    def logprob_summary(self) -> LogprobSummary | None:
        if self.logprobs is None:
            return None
        return summarize_logprobs(self.logprobs, self.masks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "input_ids": list(self.input_ids),
            "masks": self.masks.to_dict(),
            "reward": self.reward,
            "advantage": self.advantage,
            "logprobs": self.logprobs.to_dict() if self.logprobs else None,
            "logprob_summary": (
                self.logprob_summary().to_dict() if self.logprob_summary() else None
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrainingBatch:
    """A batch of sequence samples with convenience diagnostics."""

    samples: tuple[SequenceTrainingSample, ...]

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("TrainingBatch requires at least one sample")

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def mean_reward(self) -> float:
        return fmean(sample.reward for sample in self.samples)

    @property
    def total_response_tokens(self) -> int:
        return sum(sample.response_tokens for sample in self.samples)

    def by_task(self) -> dict[str, tuple[SequenceTrainingSample, ...]]:
        grouped: dict[str, list[SequenceTrainingSample]] = {}
        for sample in self.samples:
            grouped.setdefault(sample.task_id, []).append(sample)
        return {
            task_id: tuple(sorted(values, key=lambda item: item.sample_index))
            for task_id, values in sorted(grouped.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "mean_reward": self.mean_reward,
            "total_response_tokens": self.total_response_tokens,
            "samples": [sample.to_dict() for sample in self.samples],
        }
