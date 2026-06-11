"""Tokenizer contracts and prompt/response packing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, Protocol, Sequence, TypeAlias

from codeself.training.common.masking import TokenMask, build_prompt_response_mask

TruncationSide: TypeAlias = Literal["left", "right"]


@dataclass(frozen=True)
class TokenizedText:
    """A text span after tokenizer encoding and optional truncation."""

    text: str
    token_ids: tuple[int, ...]
    truncated: bool = False
    truncation_side: TruncationSide | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be non-negative integers")
        if self.truncation_side is not None:
            _validate_truncation_side(self.truncation_side)

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "token_count": self.token_count,
            "truncated": self.truncated,
            "truncation_side": self.truncation_side,
        }


@dataclass(frozen=True)
class TokenizedPromptResponse:
    """A packed prompt/response sequence with loss masks."""

    prompt: TokenizedText
    response: TokenizedText
    input_ids: tuple[int, ...]
    masks: TokenMask

    def __post_init__(self) -> None:
        expected = self.prompt.token_ids + self.response.token_ids
        if self.input_ids != expected:
            raise ValueError("input_ids must equal prompt token ids followed by response token ids")
        if len(self.input_ids) != self.masks.token_count:
            raise ValueError("input_ids length must match masks")
        if self.masks.prompt_tokens != self.prompt.token_count:
            raise ValueError("prompt mask count must match prompt token count")
        if self.masks.response_tokens != self.response.token_count:
            raise ValueError("response mask count must match response token count")

    @property
    def prompt_token_count(self) -> int:
        return self.prompt.token_count

    @property
    def response_token_count(self) -> int:
        return self.response.token_count

    @property
    def truncated(self) -> bool:
        return self.prompt.truncated or self.response.truncated

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt.to_dict(),
            "response": self.response.to_dict(),
            "input_ids": list(self.input_ids),
            "masks": self.masks.to_dict(),
            "truncated": self.truncated,
        }


class TokenizerEngine(Protocol):
    """Small tokenizer surface needed by the training core."""

    @property
    def eos_token_id(self) -> int | None:
        """Return the end-of-sequence token id when the tokenizer has one."""
        ...

    def encode(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        truncation_side: TruncationSide = "right",
    ) -> tuple[int, ...]:
        """Encode text into token ids."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token ids into text."""
        ...


class WhitespaceTokenizerEngine:
    """Deterministic dependency-free tokenizer for unit tests and smoke paths."""

    def __init__(self, *, eos_token_id: int | None = None) -> None:
        self._eos_token_id = eos_token_id
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    def encode(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        truncation_side: TruncationSide = "right",
    ) -> tuple[int, ...]:
        token_ids = tuple(self._id_for_token(token) for token in text.split())
        return truncate_token_ids(
            token_ids,
            max_tokens=max_tokens,
            truncation_side=truncation_side,
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return " ".join(
            self._id_to_token.get(int(token_id), f"<tok:{int(token_id)}>")
            for token_id in token_ids
        )

    def _id_for_token(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        candidate = _stable_token_id(token)
        while candidate in self._id_to_token and self._id_to_token[candidate] != token:
            candidate += 1
        self._token_to_id[token] = candidate
        self._id_to_token[candidate] = token
        return candidate


def truncate_token_ids(
    token_ids: Sequence[int],
    *,
    max_tokens: int | None,
    truncation_side: TruncationSide = "right",
) -> tuple[int, ...]:
    """Return token ids truncated from the right or left."""

    ids = tuple(int(token_id) for token_id in token_ids)
    _validate_truncation_side(truncation_side)
    if max_tokens is None or len(ids) <= max_tokens:
        return ids
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if truncation_side == "right":
        return ids[:max_tokens]
    return ids[-max_tokens:] if max_tokens else ()


def encode_text(
    tokenizer: TokenizerEngine,
    text: str,
    *,
    max_tokens: int | None = None,
    truncation_side: TruncationSide = "right",
) -> TokenizedText:
    """Encode text and record whether truncation changed the token sequence."""

    raw_token_ids = tuple(tokenizer.encode(text))
    token_ids = truncate_token_ids(
        raw_token_ids,
        max_tokens=max_tokens,
        truncation_side=truncation_side,
    )
    return TokenizedText(
        text=text,
        token_ids=token_ids,
        truncated=len(token_ids) != len(raw_token_ids),
        truncation_side=truncation_side if max_tokens is not None else None,
    )


def encode_prompt_response(
    tokenizer: TokenizerEngine,
    *,
    prompt: str,
    response: str,
    max_prompt_tokens: int | None = None,
    max_response_tokens: int | None = None,
    prompt_truncation_side: TruncationSide = "left",
    response_truncation_side: TruncationSide = "right",
) -> TokenizedPromptResponse:
    """Encode prompt and response as one masked training sequence."""

    prompt_tokens = encode_text(
        tokenizer,
        prompt,
        max_tokens=max_prompt_tokens,
        truncation_side=prompt_truncation_side,
    )
    response_tokens = encode_text(
        tokenizer,
        response,
        max_tokens=max_response_tokens,
        truncation_side=response_truncation_side,
    )
    input_ids = prompt_tokens.token_ids + response_tokens.token_ids
    if not input_ids:
        raise ValueError("prompt and response cannot both encode to zero tokens")
    masks = build_prompt_response_mask(
        token_count=len(input_ids),
        prompt_token_count=prompt_tokens.token_count,
        response_token_count=response_tokens.token_count,
    )
    return TokenizedPromptResponse(
        prompt=prompt_tokens,
        response=response_tokens,
        input_ids=input_ids,
        masks=masks,
    )


def _stable_token_id(token: str) -> int:
    digest = sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") + 1


def _validate_truncation_side(truncation_side: TruncationSide) -> None:
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be left or right")
