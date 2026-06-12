"""Dataset leakage and contamination checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from codeself.datasets.schemas import Split, TaskSpec


@dataclass(frozen=True)
class HiddenLeakFinding:
    """A hidden-test leakage finding for one task."""

    task_id: str
    test_name: str
    surface: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "test_name": self.test_name,
            "surface": self.surface,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class ContaminationFinding:
    """A potential train/eval contamination finding."""

    kind: str
    task_id_a: str
    split_a: str
    task_id_b: str
    split_b: str
    score: float
    field: str = "prompt"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "task_id_a": self.task_id_a,
            "split_a": self.split_a,
            "task_id_b": self.task_id_b,
            "split_b": self.split_b,
            "score": self.score,
            "field": self.field,
        }


@dataclass(frozen=True)
class DatasetQualityReport:
    """Dataset quality report for leakage and contamination checks."""

    task_count: int
    hidden_leaks: tuple[HiddenLeakFinding, ...] = field(default_factory=tuple)
    contamination_findings: tuple[ContaminationFinding, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.hidden_leaks and not self.contamination_findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "passed": self.passed,
            "hidden_leaks": [finding.to_dict() for finding in self.hidden_leaks],
            "contamination_findings": [
                finding.to_dict() for finding in self.contamination_findings
            ],
        }


def check_dataset_quality(
    tasks: list[TaskSpec],
    *,
    near_duplicate_threshold: float = 0.92,
) -> DatasetQualityReport:
    """Check task specs for hidden-test leakage and train/eval overlap."""

    return DatasetQualityReport(
        task_count=len(tasks),
        hidden_leaks=tuple(detect_hidden_test_leaks(tasks)),
        contamination_findings=tuple(
            detect_train_eval_contamination(
                tasks,
                near_duplicate_threshold=near_duplicate_threshold,
            )
        ),
    )


def detect_hidden_test_leaks(tasks: list[TaskSpec]) -> list[HiddenLeakFinding]:
    """Detect hidden tests copied into prompt, starter code, or public tests."""

    findings: list[HiddenLeakFinding] = []
    for task in tasks:
        surfaces = {
            "prompt": task.prompt,
            "starter_code": task.starter_code,
            "public_tests": "\n".join(test.code for test in task.public_tests),
        }
        for test in task.hidden_tests:
            normalized_hidden = normalize_code(test.code)
            if not normalized_hidden:
                continue
            for surface_name, surface_text in surfaces.items():
                if normalized_hidden in normalize_code(surface_text):
                    findings.append(
                        HiddenLeakFinding(
                            task_id=task.task_id,
                            test_name=test.name,
                            surface=surface_name,
                            snippet=_snippet(test.code),
                        )
                    )
    return findings


def detect_train_eval_contamination(
    tasks: list[TaskSpec],
    *,
    train_splits: tuple[Split, ...] = (Split.TRAIN,),
    eval_splits: tuple[Split, ...] = (
        Split.DEV,
        Split.TEST_PUBLIC,
        Split.TEST_PRIVATE,
        Split.STRESS,
    ),
    near_duplicate_threshold: float = 0.92,
) -> list[ContaminationFinding]:
    """Detect exact and near-duplicate prompts across train/eval boundaries."""

    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")

    train_values = [task for task in tasks if task.split in train_splits]
    eval_values = [task for task in tasks if task.split in eval_splits]
    findings: list[ContaminationFinding] = []
    for train_task in train_values:
        train_text = normalize_text(_task_surface(train_task))
        if not train_text:
            continue
        for eval_task in eval_values:
            eval_text = normalize_text(_task_surface(eval_task))
            if not eval_text:
                continue
            if train_text == eval_text:
                findings.append(
                    ContaminationFinding(
                        kind="exact_duplicate",
                        task_id_a=train_task.task_id,
                        split_a=train_task.split.value,
                        task_id_b=eval_task.task_id,
                        split_b=eval_task.split.value,
                        score=1.0,
                    )
                )
                continue
            score = SequenceMatcher(None, train_text, eval_text).ratio()
            if score >= near_duplicate_threshold:
                findings.append(
                    ContaminationFinding(
                        kind="near_duplicate",
                        task_id_a=train_task.task_id,
                        split_a=train_task.split.value,
                        task_id_b=eval_task.task_id,
                        split_b=eval_task.split.value,
                        score=score,
                    )
                )
    return findings


def normalize_text(value: str) -> str:
    """Normalize prose for conservative near-duplicate prompt checks.

    Aggressively strips punctuation so reworded prompts still match; suitable
    for prompt/prose comparison, not for code where punctuation is meaningful.
    """

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_]+", " ", value.lower())).strip()


def normalize_code(value: str) -> str:
    """Normalize code for hidden-test leak detection.

    Collapses whitespace and lowercases but *preserves punctuation*, so two
    structurally different snippets such as ``[[1], [2], [3]]`` and
    ``[[1, 2], [3]]`` are not falsely treated as identical (which happens when
    brackets and commas are stripped).
    """

    return re.sub(r"\s+", " ", value.lower()).strip()


def _task_surface(task: TaskSpec) -> str:
    return "\n".join(
        value
        for value in (
            task.prompt,
            task.starter_code,
            task.entry_point or "",
        )
        if value
    )


def _snippet(value: str, limit: int = 80) -> str:
    stripped = " ".join(value.split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."
