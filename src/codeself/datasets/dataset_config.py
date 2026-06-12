"""Versioned dataset configuration and build pipeline.

A dataset config declares the full task distribution for an experiment in one
versioned file: which sources to ingest, how visible/hidden tests are split,
how splits are assigned, and the contamination thresholds that gate the build.

Building a dataset is deterministic given the config (sources + seed), and the
build *fails hard* when contamination or hidden-test leakage is detected so a
contaminated dataset can never silently feed training or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from codeself.config import load_config_file
from codeself.datasets.contamination import DatasetQualityReport, check_dataset_quality
from codeself.datasets.loaders import (
    apply_test_sidecar,
    load_humaneval_like,
    load_mbpp,
    load_task_specs,
)
from codeself.datasets.schemas import Split, TaskSpec, TestSpec, TestVisibility
from codeself.datasets.splits import (
    SplitFractions,
    assign_splits,
    dataset_fingerprint,
    split_counts,
)


class DatasetConfigError(ValueError):
    """Raised when a dataset config is malformed."""


class DatasetContaminationError(RuntimeError):
    """Raised when a dataset build fails its contamination/leakage gate."""

    def __init__(self, report: DatasetQualityReport) -> None:
        self.report = report
        leak_count = len(report.hidden_leaks)
        contam_count = len(report.contamination_findings)
        super().__init__(
            f"dataset failed quality gate: {leak_count} hidden-test leak(s), "
            f"{contam_count} contamination finding(s)"
        )


@dataclass(frozen=True)
class DatasetSourceConfig:
    """One ingestion source within a dataset config."""

    format: str
    path: str
    source: str | None = None
    split: str | None = None
    sidecar: str | None = None
    public_tests_per_task: int = 1

    def __post_init__(self) -> None:
        if self.format not in {"mbpp", "humaneval", "canonical"}:
            raise DatasetConfigError(f"unknown source format: {self.format}")
        if not self.path:
            raise DatasetConfigError("source path must not be empty")
        if self.public_tests_per_task < 1:
            raise DatasetConfigError("public_tests_per_task must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "path": self.path,
            "source": self.source,
            "split": self.split,
            "sidecar": self.sidecar,
            "public_tests_per_task": self.public_tests_per_task,
        }


@dataclass(frozen=True)
class DatasetConfig:
    """Versioned declaration of a task distribution."""

    name: str
    version: str
    sources: tuple[DatasetSourceConfig, ...]
    seed: int = 20260601
    assign_splits: bool = True
    split_fractions: SplitFractions = field(default_factory=SplitFractions)
    auto_split_hidden: bool = True
    visible_tests_per_task: int = 1
    near_duplicate_threshold: float = 0.92
    enforce_quality_gate: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise DatasetConfigError("dataset name must not be empty")
        if not self.version.strip():
            raise DatasetConfigError("dataset version must not be empty")
        if not self.sources:
            raise DatasetConfigError("dataset config must declare at least one source")
        if self.visible_tests_per_task < 1:
            raise DatasetConfigError("visible_tests_per_task must be at least 1")
        if not 0 < self.near_duplicate_threshold <= 1:
            raise DatasetConfigError("near_duplicate_threshold must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "assign_splits": self.assign_splits,
            "split_fractions": {
                "train": self.split_fractions.train,
                "dev": self.split_fractions.dev,
                "test_public": self.split_fractions.test_public,
                "test_private": self.split_fractions.test_private,
            },
            "auto_split_hidden": self.auto_split_hidden,
            "visible_tests_per_task": self.visible_tests_per_task,
            "near_duplicate_threshold": self.near_duplicate_threshold,
            "enforce_quality_gate": self.enforce_quality_gate,
            "sources": [source.to_dict() for source in self.sources],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: Path | None = None) -> "DatasetConfig":
        dataset = value.get("dataset", value)
        if not isinstance(dataset, dict):
            raise DatasetConfigError("dataset config must be a mapping")
        raw_sources = dataset.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise DatasetConfigError("dataset config must declare a non-empty sources list")
        sources = tuple(_source_from_dict(item, base_dir=base_dir) for item in raw_sources)
        fractions = _fractions_from_dict(dataset.get("split_fractions"))
        return cls(
            name=str(dataset.get("name", "")),
            version=str(dataset.get("version", "")),
            sources=sources,
            seed=int(dataset.get("seed", 20260601)),
            assign_splits=bool(dataset.get("assign_splits", True)),
            split_fractions=fractions,
            auto_split_hidden=bool(dataset.get("auto_split_hidden", True)),
            visible_tests_per_task=int(dataset.get("visible_tests_per_task", 1)),
            near_duplicate_threshold=float(dataset.get("near_duplicate_threshold", 0.92)),
            enforce_quality_gate=bool(dataset.get("enforce_quality_gate", True)),
        )


@dataclass(frozen=True)
class DatasetBuildResult:
    """Outcome of building a dataset from a config."""

    config: DatasetConfig
    tasks: tuple[TaskSpec, ...]
    quality_report: DatasetQualityReport
    auto_split_task_ids: tuple[str, ...]
    tasks_without_hidden: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        return dataset_fingerprint(list(self.tasks))

    def manifest(self) -> dict[str, Any]:
        return {
            "dataset": {
                "name": self.config.name,
                "version": self.config.version,
                "seed": self.config.seed,
                "task_count": len(self.tasks),
                "fingerprint": self.fingerprint,
            },
            "config": self.config.to_dict(),
            "splits": split_counts(list(self.tasks)),
            "auto_split_task_count": len(self.auto_split_task_ids),
            "tasks_without_hidden": list(self.tasks_without_hidden),
            "quality_report": self.quality_report.to_dict(),
        }


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load a dataset config from a JSON or simple-YAML file."""

    config_path = Path(path)
    loaded = load_config_file(config_path)
    return DatasetConfig.from_dict(loaded, base_dir=config_path.parent)


def build_dataset(config: DatasetConfig) -> DatasetBuildResult:
    """Build the declared task distribution and enforce the quality gate.

    Raises ``DatasetContaminationError`` when ``enforce_quality_gate`` is set
    and contamination or hidden-test leakage is detected.
    """

    tasks: list[TaskSpec] = []
    for source in config.sources:
        tasks.extend(_load_source(source))

    auto_split_ids: list[str] = []
    if config.auto_split_hidden:
        tasks, auto_split_ids = _auto_split_hidden_tests(
            tasks,
            visible_tests_per_task=config.visible_tests_per_task,
        )

    if config.assign_splits:
        tasks = assign_splits(tasks, fractions=config.split_fractions, seed=config.seed)

    tasks = sorted(tasks, key=lambda task: task.task_id)
    _ensure_unique_task_ids(tasks)

    report = check_dataset_quality(
        tasks,
        near_duplicate_threshold=config.near_duplicate_threshold,
    )
    if config.enforce_quality_gate and not report.passed:
        raise DatasetContaminationError(report)

    tasks_without_hidden = tuple(task.task_id for task in tasks if not task.hidden_tests)
    return DatasetBuildResult(
        config=config,
        tasks=tuple(tasks),
        quality_report=report,
        auto_split_task_ids=tuple(auto_split_ids),
        tasks_without_hidden=tasks_without_hidden,
    )


def auto_split_visible_hidden(
    task: TaskSpec,
    *,
    visible_tests_per_task: int = 1,
) -> tuple[TaskSpec, bool]:
    """Hold out a deterministic slice of tests as hidden when none exist.

    The model may see ``visible_tests_per_task`` example tests; the remainder
    become held-out hidden tests that drive reward and evaluation. Tasks that
    already declare hidden tests, or that have too few tests to split, are
    returned unchanged.
    """

    if task.hidden_tests:
        return task, False
    if len(task.public_tests) <= visible_tests_per_task:
        return task, False

    visible = task.public_tests[:visible_tests_per_task]
    held_out = task.public_tests[visible_tests_per_task:]
    hidden = tuple(
        TestSpec(
            name=_hidden_name(test.name),
            code=test.code,
            visibility=TestVisibility.HIDDEN,
            kind=test.kind,
        )
        for test in held_out
    )
    return replace(task, public_tests=visible, hidden_tests=hidden), True


def _auto_split_hidden_tests(
    tasks: list[TaskSpec],
    *,
    visible_tests_per_task: int,
) -> tuple[list[TaskSpec], list[str]]:
    updated: list[TaskSpec] = []
    split_ids: list[str] = []
    for task in tasks:
        new_task, was_split = auto_split_visible_hidden(
            task,
            visible_tests_per_task=visible_tests_per_task,
        )
        updated.append(new_task)
        if was_split:
            split_ids.append(task.task_id)
    return updated, split_ids


def _load_source(source: DatasetSourceConfig) -> list[TaskSpec]:
    source_label = source.source or source.format
    if source.format == "mbpp":
        tasks = load_mbpp(
            source.path,
            split=source.split or Split.TRAIN.value,
            source=source_label,
            public_tests_per_task=source.public_tests_per_task,
        )
    elif source.format == "humaneval":
        tasks = load_humaneval_like(
            source.path,
            split=source.split or Split.TEST_PUBLIC.value,
            source=source_label,
        )
    else:
        tasks = load_task_specs(source.path)
        if source.split is not None:
            split = Split(source.split)
            tasks = [replace(task, split=split) for task in tasks]
    if source.sidecar:
        tasks = apply_test_sidecar(tasks, source.sidecar)
    return tasks


def _ensure_unique_task_ids(tasks: list[TaskSpec]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise DatasetConfigError(f"duplicate task_id across sources: {task.task_id}")
        seen.add(task.task_id)


def _hidden_name(name: str) -> str:
    if name.startswith("public"):
        return "hidden" + name[len("public"):]
    return f"hidden-{name}"


def _source_from_dict(value: Any, *, base_dir: Path | None) -> DatasetSourceConfig:
    if not isinstance(value, dict):
        raise DatasetConfigError("each source must be a mapping")
    path = str(value.get("path", ""))
    if base_dir is not None and path and not Path(path).is_absolute():
        path = str((base_dir / path).resolve())
    sidecar = value.get("sidecar")
    if base_dir is not None and sidecar and not Path(str(sidecar)).is_absolute():
        sidecar = str((base_dir / str(sidecar)).resolve())
    return DatasetSourceConfig(
        format=str(value.get("format", "")),
        path=path,
        source=_optional_str(value.get("source")),
        split=_optional_str(value.get("split")),
        sidecar=_optional_str(sidecar),
        public_tests_per_task=int(value.get("public_tests_per_task", 1)),
    )


def _fractions_from_dict(value: Any) -> SplitFractions:
    if value is None:
        return SplitFractions()
    if not isinstance(value, dict):
        raise DatasetConfigError("split_fractions must be a mapping")
    return SplitFractions(
        train=float(value.get("train", 0.80)),
        dev=float(value.get("dev", 0.10)),
        test_public=float(value.get("test_public", 0.10)),
        test_private=float(value.get("test_private", 0.0)),
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
