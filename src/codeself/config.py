"""Dependency-free experiment config loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ConfigLoadError(ValueError):
    """Raised when an experiment config cannot be parsed."""


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or simple YAML config file.

    The YAML parser intentionally supports only the small subset used by the
    repository configs: nested mappings, scalar values, and inline lists.
    """

    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigLoadError(f"invalid JSON config: {input_path}") from exc
    elif input_path.suffix.lower() in {".yaml", ".yml"}:
        loaded = _parse_simple_yaml(text)
    else:
        raise ConfigLoadError(f"unsupported config suffix: {input_path.suffix}")
    if not isinstance(loaded, dict):
        raise ConfigLoadError("experiment config must be a mapping")
    return loaded


def config_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    """Return a dotted-path value from a nested config mapping."""

    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    lines = _yaml_lines(text)
    if not lines:
        return {}
    parsed, index = _parse_yaml_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ConfigLoadError("could not parse entire YAML config")
    if not isinstance(parsed, dict):
        raise ConfigLoadError("YAML config root must be a mapping")
    return parsed


def _yaml_lines(text: str) -> list[tuple[int, str]]:
    parsed: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        without_comment = _strip_yaml_comment(raw_line).rstrip()
        if not without_comment.strip():
            continue
        indent = len(without_comment) - len(without_comment.lstrip(" "))
        if indent % 2 != 0:
            raise ConfigLoadError(f"YAML indentation must use multiples of two at line {line_number}")
        parsed.append((indent, without_comment.strip()))
    return parsed


def _parse_yaml_block(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_mapping(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ConfigLoadError(f"unexpected indentation before {content!r}")
        if content.startswith("- "):
            break
        key, value_text = _split_yaml_key_value(content)
        index += 1
        if value_text == "":
            if index < len(lines) and lines[index][0] > line_indent:
                value, index = _parse_yaml_block(lines, index, lines[index][0])
            else:
                value = {}
        else:
            value = _parse_yaml_scalar(value_text)
        result[key] = value
    return result, index


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    result: list[Any] = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ConfigLoadError(f"unexpected indentation before {content!r}")
        if not content.startswith("- "):
            break
        item_text = content[2:].strip()
        content_indent = line_indent + 2
        index += 1

        # Collect any deeper-indented lines belonging to this item, so a
        # list entry can be a multi-key block mapping (standard YAML).
        item_lines: list[tuple[int, str]] = [(content_indent, item_text)] if item_text else []
        while index < len(lines) and lines[index][0] > line_indent:
            item_lines.append(lines[index])
            index += 1

        if not item_lines:
            value: Any = None
        elif (
            len(item_lines) == 1
            and not _looks_like_key_value(item_text)
            and not item_text.startswith("- ")
        ):
            value = _parse_yaml_scalar(item_text)
        else:
            value, _ = _parse_yaml_block(item_lines, 0, content_indent)
        result.append(value)
    return result, index


def _split_yaml_key_value(content: str) -> tuple[str, str]:
    if ":" not in content:
        raise ConfigLoadError(f"expected YAML key/value pair, got {content!r}")
    key, value = content.split(":", 1)
    key = key.strip()
    if not key:
        raise ConfigLoadError("YAML key must not be empty")
    return key, value.strip()


def _parse_yaml_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return [_parse_yaml_scalar(item.strip()) for item in _split_inline_list(value[1:-1])]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?\d*\.\d+", value):
        return float(value)
    return value


def _split_inline_list(value: str) -> list[str]:
    if not value.strip():
        return []
    return [item.strip() for item in value.split(",")]


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _looks_like_key_value(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_-]+\s*:", value))
