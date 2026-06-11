"""Prompt/response token mask helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TokenMask:
    """Prompt, response, and attention masks for one token sequence."""

    attention_mask: tuple[int, ...]
    prompt_mask: tuple[int, ...]
    response_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = {len(self.attention_mask), len(self.prompt_mask), len(self.response_mask)}
        if len(lengths) != 1:
            raise ValueError("all masks must have the same length")
        for mask in (self.attention_mask, self.prompt_mask, self.response_mask):
            if any(value not in {0, 1} for value in mask):
                raise ValueError("masks must contain only 0/1 values")
        if any(p and r for p, r in zip(self.prompt_mask, self.response_mask, strict=True)):
            raise ValueError("prompt and response masks must not overlap")

    @property
    def token_count(self) -> int:
        return len(self.attention_mask)

    @property
    def prompt_tokens(self) -> int:
        return sum(self.prompt_mask)

    @property
    def response_tokens(self) -> int:
        return sum(self.response_mask)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "attention_mask": list(self.attention_mask),
            "prompt_mask": list(self.prompt_mask),
            "response_mask": list(self.response_mask),
        }


def build_prompt_response_mask(
    *,
    token_count: int,
    prompt_token_count: int,
    response_token_count: int | None = None,
    pad_token_count: int = 0,
) -> TokenMask:
    """Build non-overlapping prompt/response masks for a padded sequence."""

    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if prompt_token_count < 0 or pad_token_count < 0:
        raise ValueError("prompt_token_count and pad_token_count must be non-negative")
    if prompt_token_count + pad_token_count > token_count:
        raise ValueError("prompt and padding exceed token_count")
    active_tokens = token_count - pad_token_count
    if response_token_count is None:
        response_token_count = active_tokens - prompt_token_count
    if response_token_count < 0:
        raise ValueError("response_token_count must be non-negative")
    if prompt_token_count + response_token_count > active_tokens:
        raise ValueError("prompt and response exceed non-padding tokens")

    prompt_mask = [0] * token_count
    response_mask = [0] * token_count
    attention_mask = [1] * active_tokens + [0] * pad_token_count
    for index in range(prompt_token_count):
        prompt_mask[index] = 1
    response_start = prompt_token_count
    for index in range(response_start, response_start + response_token_count):
        response_mask[index] = 1
    return TokenMask(
        attention_mask=tuple(attention_mask),
        prompt_mask=tuple(prompt_mask),
        response_mask=tuple(response_mask),
    )


def mask_sum(values: Iterable[float], mask: Iterable[int]) -> float:
    """Sum values where mask is 1."""

    value_items = tuple(values)
    mask_items = tuple(mask)
    if len(value_items) != len(mask_items):
        raise ValueError("values and mask must have equal length")
    return sum(value for value, include in zip(value_items, mask_items, strict=True) if include)


def mask_mean(values: Iterable[float], mask: Iterable[int]) -> float:
    """Mean of values where mask is 1."""

    mask_items = tuple(mask)
    count = sum(mask_items)
    if count == 0:
        return 0.0
    return mask_sum(values, mask_items) / count
