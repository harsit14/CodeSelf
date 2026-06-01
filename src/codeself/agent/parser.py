"""Parse executable code from model completions."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum


class ParseStatus(str, Enum):
    """Parser result status."""

    OK = "ok"
    EMPTY = "empty"
    SYNTAX_ERROR = "syntax_error"


@dataclass(frozen=True)
class ParsedCompletion:
    """Parsed model completion."""

    raw_text: str
    code: str
    status: ParseStatus
    parser: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == ParseStatus.OK


FENCE_RE = re.compile(r"```(?P<lang>[A-Za-z0-9_+-]*)\s*\n(?P<code>.*?)```", re.DOTALL)


def extract_code(completion: str) -> ParsedCompletion:
    """Extract the most plausible Python code block from a completion."""

    raw = completion or ""
    candidates = _fenced_candidates(raw)
    if not candidates:
        candidates = [("raw", _strip_common_preamble(raw))]

    best_error = ""
    for parser_name, candidate in candidates:
        cleaned = _clean_candidate(candidate)
        if not cleaned:
            continue
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            best_error = f"{exc.msg} at line {exc.lineno}"
            continue
        return ParsedCompletion(raw_text=raw, code=cleaned, status=ParseStatus.OK, parser=parser_name)

    fallback = _clean_candidate(candidates[0][1]) if candidates else ""
    if not fallback:
        return ParsedCompletion(raw_text=raw, code="", status=ParseStatus.EMPTY, parser="empty")
    return ParsedCompletion(
        raw_text=raw,
        code=fallback,
        status=ParseStatus.SYNTAX_ERROR,
        parser=candidates[0][0],
        error=best_error or "no syntactically valid Python code found",
    )


def _fenced_candidates(text: str) -> list[tuple[str, str]]:
    matches = list(FENCE_RE.finditer(text))
    python_blocks: list[tuple[str, str]] = []
    other_blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches, start=1):
        language = match.group("lang").lower()
        candidate = match.group("code")
        if language in {"python", "py", ""}:
            python_blocks.append((f"fenced:{language or 'plain'}:{index}", candidate))
        else:
            other_blocks.append((f"fenced:{language}:{index}", candidate))
    return python_blocks or other_blocks


def _strip_common_preamble(text: str) -> str:
    stripped = text.strip()
    markers = ("def ", "class ", "import ", "from ")
    positions = [stripped.find(marker) for marker in markers if stripped.find(marker) >= 0]
    if positions:
        return stripped[min(positions) :]
    return stripped


def _clean_candidate(candidate: str) -> str:
    lines = candidate.strip().splitlines()
    while lines and lines[0].strip().lower() in {"python", "py"}:
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    return cleaned + "\n" if cleaned else ""
