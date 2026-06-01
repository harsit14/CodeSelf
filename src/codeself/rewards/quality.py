"""Lightweight code-quality diagnostics.

These metrics are logged for the MVP reward but are not optimized directly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class QualityMetrics:
    """Simple quality metrics for a generated Python solution."""

    syntax_valid: bool
    line_count: int
    non_empty_line_count: int
    char_count: int
    max_line_length: int
    quality_score: float

    @classmethod
    def empty(cls) -> "QualityMetrics":
        return cls(
            syntax_valid=False,
            line_count=0,
            non_empty_line_count=0,
            char_count=0,
            max_line_length=0,
            quality_score=0.0,
        )

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "syntax_valid": self.syntax_valid,
            "line_count": self.line_count,
            "non_empty_line_count": self.non_empty_line_count,
            "char_count": self.char_count,
            "max_line_length": self.max_line_length,
            "score": self.quality_score,
        }


def measure_quality(code: str) -> QualityMetrics:
    """Return simple, dependency-free quality diagnostics."""

    lines = code.splitlines()
    non_empty = [line for line in lines if line.strip()]
    max_line_length = max((len(line) for line in lines), default=0)
    syntax_valid = _syntax_valid(code)
    score = _quality_score(
        syntax_valid=syntax_valid,
        non_empty_line_count=len(non_empty),
        char_count=len(code),
        max_line_length=max_line_length,
    )
    return QualityMetrics(
        syntax_valid=syntax_valid,
        line_count=len(lines),
        non_empty_line_count=len(non_empty),
        char_count=len(code),
        max_line_length=max_line_length,
        quality_score=score,
    )


def _syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _quality_score(
    *,
    syntax_valid: bool,
    non_empty_line_count: int,
    char_count: int,
    max_line_length: int,
) -> float:
    if not syntax_valid:
        return 0.0
    score = 1.0
    if max_line_length > 100:
        score -= 0.15
    if non_empty_line_count > 120:
        score -= 0.20
    if char_count > 8_000:
        score -= 0.20
    return max(0.0, min(1.0, score))
