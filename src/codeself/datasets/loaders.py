"""Dataset loaders for local benchmark files."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from codeself.datasets.schemas import Split, TaskSpec, TestSpec, TestVisibility


class DatasetLoadError(ValueError):
    """Raised when a dataset file cannot be converted into TaskSpec records."""


def load_mbpp(
    path: str | Path,
    *,
    split: Split | str = Split.TRAIN,
    source: str = "mbpp",
    public_tests_per_task: int = 1,
) -> list[TaskSpec]:
    """Load an MBPP-style JSON/JSONL file into canonical task specs.

    Supported fields include the common MBPP names:
    `task_id`, `text`, `code`, `test_list`, and `challenge_test_list`.
    """

    tasks: list[TaskSpec] = []
    for record in _read_records(path):
        raw_tests = _string_list(record.get("test_list", []))
        challenge_tests = _string_list(record.get("challenge_test_list", []))
        public_tests = raw_tests[:public_tests_per_task]
        hidden_tests = raw_tests[public_tests_per_task:] + challenge_tests
        task_id = _scoped_task_id(source, record.get("task_id", len(tasks)))
        entry_point = _entry_point(record, raw_tests + challenge_tests)
        prompt = str(record.get("text") or record.get("prompt") or "").strip()
        if not prompt:
            raise DatasetLoadError(f"MBPP record {task_id} is missing text/prompt")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                source=source,
                prompt=prompt,
                split=split,
                entry_point=entry_point,
                starter_code=str(record.get("starter_code", "")),
                public_tests=_tests(public_tests, TestVisibility.PUBLIC, prefix="public"),
                hidden_tests=_tests(hidden_tests, TestVisibility.HIDDEN, prefix="hidden"),
                tags=tuple(_string_list(record.get("tags", []))),
                metadata=_metadata(record, include=("code", "difficulty")),
            )
        )
    return tasks


def load_humaneval_like(
    path: str | Path,
    *,
    split: Split | str = Split.TEST_PUBLIC,
    source: str = "humaneval",
) -> list[TaskSpec]:
    """Load HumanEval/EvalPlus-style JSONL records into canonical task specs."""

    tasks: list[TaskSpec] = []
    for record in _read_records(path):
        raw_task_id = record.get("task_id", len(tasks))
        task_id = _scoped_task_id(source, raw_task_id)
        prompt = str(record.get("prompt") or "").strip()
        if not prompt:
            raise DatasetLoadError(f"HumanEval-like record {task_id} is missing prompt")

        public_tests = _string_list(
            record.get("public_tests")
            or record.get("visible_tests")
            or record.get("example_tests")
            or []
        )
        entry_point = _entry_point(record, public_tests)
        hidden_tests = _hidden_tests_from_humaneval_record(record, entry_point)
        entry_point = entry_point or _entry_point(record, public_tests + hidden_tests)

        tasks.append(
            TaskSpec(
                task_id=task_id,
                source=source,
                prompt=prompt,
                split=split,
                entry_point=entry_point,
                starter_code=str(record.get("starter_code", "")),
                public_tests=_tests(public_tests, TestVisibility.PUBLIC, prefix="public"),
                hidden_tests=_tests(hidden_tests, TestVisibility.HIDDEN, prefix="hidden"),
                tags=tuple(_string_list(record.get("tags", []))),
                metadata=_metadata(record, include=("canonical_solution", "difficulty")),
            )
        )
    return tasks


def load_task_specs(path: str | Path) -> list[TaskSpec]:
    """Load canonical CodeSelf task JSONL."""

    tasks: list[TaskSpec] = []
    for record in _read_records(path):
        tasks.append(TaskSpec.from_dict(record))
    return tasks


def apply_test_sidecar(tasks: Iterable[TaskSpec], sidecar_path: str | Path) -> list[TaskSpec]:
    """Apply public/hidden tests from a sidecar file keyed by task ID."""

    sidecar = load_test_sidecar(sidecar_path)
    updated: list[TaskSpec] = []
    for task in tasks:
        replacement = sidecar.get(task.task_id) or sidecar.get(_unscoped_task_id(task.task_id))
        if replacement is None:
            updated.append(task)
            continue
        public_tests = replacement.get("public_tests", task.public_tests)
        hidden_tests = replacement.get("hidden_tests", task.hidden_tests)
        updated.append(replace(task, public_tests=public_tests, hidden_tests=hidden_tests))
    return updated


def load_test_sidecar(path: str | Path) -> dict[str, dict[str, tuple[TestSpec, ...]]]:
    """Load sidecar tests.

    Supported JSON object format:

    ```json
    {
      "task/id": {
        "public_tests": [{"name": "public", "code": "assert f() == 1"}],
        "hidden_tests": [{"name": "hidden", "code": "assert f() == 2"}]
      }
    }
    ```

    JSONL records with a `task_id` field are also accepted.
    """

    loaded = _read_json(path)
    if isinstance(loaded, dict) and not _looks_like_record(loaded):
        records = [dict(value, task_id=key) for key, value in loaded.items()]
    else:
        records = _records_from_loaded(loaded)

    sidecar: dict[str, dict[str, tuple[TestSpec, ...]]] = {}
    for record in records:
        task_id = str(record["task_id"])
        public_items = record.get("public_tests", [])
        hidden_items = record.get("hidden_tests", record.get("tests", []))
        sidecar[task_id] = {
            "public_tests": _sidecar_tests(public_items, TestVisibility.PUBLIC, "public"),
            "hidden_tests": _sidecar_tests(hidden_items, TestVisibility.HIDDEN, "hidden"),
        }
    return sidecar


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    return _records_from_loaded(_read_json(path))


def _read_json(path: str | Path) -> Any:
    input_path = Path(path)
    if input_path.suffix == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise DatasetLoadError(f"invalid JSON at {input_path}:{line_number}") from exc
        return records

    with input_path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(f"invalid JSON file: {input_path}") from exc


def _records_from_loaded(loaded: Any) -> list[dict[str, Any]]:
    if isinstance(loaded, list):
        return [_ensure_record(item) for item in loaded]
    if isinstance(loaded, dict):
        for key in ("tasks", "data", "items", "examples", "problems"):
            value = loaded.get(key)
            if isinstance(value, list):
                return [_ensure_record(item) for item in value]
        if _looks_like_record(loaded):
            return [loaded]
    raise DatasetLoadError("dataset must be a JSON list, JSONL file, or object with task records")


def _ensure_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetLoadError("each dataset item must be an object")
    return value


def _looks_like_record(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("task_id", "prompt", "text", "test_list", "test"))


def _hidden_tests_from_humaneval_record(
    record: Mapping[str, Any],
    entry_point: str | None,
) -> list[str]:
    raw_tests: list[str] = []
    for key in ("hidden_tests", "tests", "test_list"):
        raw_tests.extend(_string_list(record.get(key, [])))
    if isinstance(record.get("test"), str):
        raw_tests.append(_maybe_call_humaneval_check(str(record["test"]), entry_point))
    return raw_tests


def _tests(values: list[str], visibility: TestVisibility, *, prefix: str) -> tuple[TestSpec, ...]:
    return tuple(
        TestSpec(name=f"{prefix}-{index}", code=value, visibility=visibility)
        for index, value in enumerate(values, start=1)
        if value.strip()
    )


def _sidecar_tests(
    values: Any,
    visibility: TestVisibility,
    prefix: str,
) -> tuple[TestSpec, ...]:
    if isinstance(values, str):
        values = [values]
    tests: list[TestSpec] = []
    for index, value in enumerate(values, start=1):
        if isinstance(value, dict):
            tests.append(TestSpec.from_dict({**value, "visibility": visibility.value}))
        elif isinstance(value, str) and value.strip():
            tests.append(TestSpec(name=f"{prefix}-{index}", code=value, visibility=visibility))
        else:
            raise DatasetLoadError("sidecar tests must be strings or objects")
    return tuple(tests)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _scoped_task_id(source: str, raw_task_id: Any) -> str:
    text = str(raw_task_id).strip()
    if "/" in text:
        return text
    return f"{source}/{text}"


def _unscoped_task_id(task_id: str) -> str:
    return task_id.rsplit("/", 1)[-1]


def _entry_point(record: Mapping[str, Any], tests: list[str]) -> str | None:
    if record.get("entry_point"):
        return str(record["entry_point"])
    for test in tests:
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match:
            return match.group(1)
        for candidate in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test):
            if candidate not in {"assertEqual", "assertTrue", "assertFalse", "print", "range"}:
                return candidate
    return None


def _maybe_call_humaneval_check(test_code: str, entry_point: str | None) -> str:
    if not entry_point:
        return test_code
    if "def check(" not in test_code:
        return test_code
    if f"check({entry_point})" in test_code:
        return test_code
    return f"{test_code.rstrip()}\n\ncheck({entry_point})"


def _metadata(record: Mapping[str, Any], *, include: tuple[str, ...]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in include:
        if key in record and record[key] is not None:
            metadata[key] = str(record[key])
    return metadata
