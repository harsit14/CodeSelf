"""Build PPO training batches from generated execution rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from codeself.agent.parser import ParseStatus
from codeself.agent.rollouts import RolloutRecord
from codeself.training.common import (
    SequenceTrainingSample,
    TokenizedPromptResponse,
    TokenizerEngine,
    TrainingBatch,
    encode_prompt_response,
)
from codeself.training.common.tokenization import TruncationSide
from codeself.training.grpo_rollouts import SkippedRollout

ResponseSource = Literal["raw_completion", "parsed_code"]


@dataclass(frozen=True)
class PPORolloutBatchConfig:
    """Settings for converting execution rollouts into PPO samples."""

    max_prompt_tokens: int | None = None
    max_response_tokens: int | None = None
    response_source: ResponseSource = "raw_completion"
    include_failed_parses: bool = True
    prompt_truncation_side: TruncationSide = "left"
    response_truncation_side: TruncationSide = "right"

    def __post_init__(self) -> None:
        _validate_optional_positive("max_prompt_tokens", self.max_prompt_tokens)
        _validate_optional_positive("max_response_tokens", self.max_response_tokens)
        if self.response_source not in {"raw_completion", "parsed_code"}:
            raise ValueError("response_source must be raw_completion or parsed_code")
        if self.prompt_truncation_side not in {"left", "right"}:
            raise ValueError("prompt_truncation_side must be left or right")
        if self.response_truncation_side not in {"left", "right"}:
            raise ValueError("response_truncation_side must be left or right")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_response_tokens": self.max_response_tokens,
            "response_source": self.response_source,
            "include_failed_parses": self.include_failed_parses,
            "prompt_truncation_side": self.prompt_truncation_side,
            "response_truncation_side": self.response_truncation_side,
        }


@dataclass(frozen=True)
class PPORolloutBatchResult:
    """A PPO-ready batch plus rollout conversion diagnostics."""

    batch: TrainingBatch
    skipped: tuple[SkippedRollout, ...]
    records_seen: int
    records_used: int
    config: PPORolloutBatchConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "batch": self.batch.to_dict(),
            "skipped": [rollout.to_dict() for rollout in self.skipped],
            "records_seen": self.records_seen,
            "records_used": self.records_used,
            "config": self.config.to_dict(),
        }


def build_ppo_training_batch_from_rollouts(
    records: Iterable[RolloutRecord],
    tokenizer: TokenizerEngine,
    *,
    config: PPORolloutBatchConfig | None = None,
) -> PPORolloutBatchResult:
    """Tokenize rollout prompt/responses into PPO training samples."""

    batch_config = config or PPORolloutBatchConfig()
    samples: list[SequenceTrainingSample] = []
    skipped: list[SkippedRollout] = []
    records_seen = 0

    for record in records:
        records_seen += 1
        sample = _build_sample(record, tokenizer, batch_config)
        if isinstance(sample, SkippedRollout):
            skipped.append(sample)
            continue
        samples.append(sample)

    if not samples:
        raise ValueError("no rollout records could be converted into PPO samples")

    return PPORolloutBatchResult(
        batch=TrainingBatch(tuple(samples)),
        skipped=tuple(skipped),
        records_seen=records_seen,
        records_used=len(samples),
        config=batch_config,
    )


def _build_sample(
    record: RolloutRecord,
    tokenizer: TokenizerEngine,
    config: PPORolloutBatchConfig,
) -> SequenceTrainingSample | SkippedRollout:
    if not config.include_failed_parses and record.parsed.status != ParseStatus.OK:
        return _skip(record, f"parse_status_{record.parsed.status.value}")

    response_text = _select_response_text(record, config.response_source)
    if not response_text.strip():
        return _skip(record, "empty_response_text")

    try:
        reward = float(record.reward.get("reward", 0.0))
    except (TypeError, ValueError):
        return _skip(record, "invalid_reward")

    packed = encode_prompt_response(
        tokenizer,
        prompt=record.prompt,
        response=response_text,
        max_prompt_tokens=config.max_prompt_tokens,
        max_response_tokens=config.max_response_tokens,
        prompt_truncation_side=config.prompt_truncation_side,
        response_truncation_side=config.response_truncation_side,
    )
    if packed.response_token_count == 0:
        return _skip(record, "empty_response_tokens")

    return SequenceTrainingSample(
        task_id=record.task_id,
        sample_index=record.sample_index,
        input_ids=packed.input_ids,
        masks=packed.masks,
        reward=reward,
        metadata=_sample_metadata(record, packed, config.response_source),
    )


def _select_response_text(record: RolloutRecord, response_source: ResponseSource) -> str:
    if response_source == "raw_completion":
        return record.raw_completion
    return record.parsed.code


def _sample_metadata(
    record: RolloutRecord,
    packed: TokenizedPromptResponse,
    response_source: ResponseSource,
) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "backend": record.backend,
        "model_name": record.model_name,
        "prompt_template": record.prompt_template,
        "parse_status": record.parsed.status.value,
        "parser": record.parsed.parser,
        "response_source": response_source,
        "prompt_tokens": packed.prompt_token_count,
        "response_tokens": packed.response_token_count,
        "prompt_truncated": packed.prompt.truncated,
        "response_truncated": packed.response.truncated,
    }
    execution_passed = record.execution.get("passed")
    if isinstance(execution_passed, bool):
        metadata["execution_passed"] = execution_passed
    reward_name = record.reward.get("reward_name")
    if isinstance(reward_name, str):
        metadata["reward_name"] = reward_name
    _copy_scalar_metadata(metadata, "generation", record.generation_metadata)
    _copy_scalar_metadata(metadata, "rollout", record.metadata)
    return metadata


def _copy_scalar_metadata(
    target: dict[str, str | int | float | bool],
    prefix: str,
    source: dict[str, object],
) -> None:
    for key, value in source.items():
        if isinstance(key, str) and isinstance(value, (str, int, float, bool)):
            target[f"{prefix}_{key}"] = value


def _skip(record: RolloutRecord, reason: str) -> SkippedRollout:
    return SkippedRollout(
        task_id=record.task_id,
        sample_index=record.sample_index,
        reason=reason,
    )


def _validate_optional_positive(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when set")
