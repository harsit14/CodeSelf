"""Core task schema for coding-agent experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Split(str, Enum):
    """Experiment split names used throughout the project."""

    TRAIN = "train"
    DEV = "dev"
    TEST_PUBLIC = "test_public"
    TEST_PRIVATE = "test_private"
    STRESS = "stress"


class TestVisibility(str, Enum):
    """Whether a test may be exposed to the agent."""

    PUBLIC = "public"
    HIDDEN = "hidden"


@dataclass(frozen=True)
class ResourceLimits:
    """Execution limits for a generated solution."""

    timeout_seconds: float = 2.0
    memory_mb: int = 256

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.memory_mb <= 0:
            raise ValueError("memory_mb must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "memory_mb": self.memory_mb,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "ResourceLimits":
        if value is None:
            return cls()
        return cls(
            timeout_seconds=float(value.get("timeout_seconds", 2.0)),
            memory_mb=int(value.get("memory_mb", 256)),
        )


@dataclass(frozen=True)
class TestSpec:
    """A test snippet associated with a task."""

    name: str
    code: str
    visibility: TestVisibility | str = TestVisibility.HIDDEN
    kind: str = "unit"

    def __post_init__(self) -> None:
        if isinstance(self.visibility, str):
            object.__setattr__(self, "visibility", TestVisibility(self.visibility))
        if not self.name.strip():
            raise ValueError("test name must not be empty")
        if not self.code.strip():
            raise ValueError("test code must not be empty")
        if not self.kind.strip():
            raise ValueError("test kind must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "code": self.code,
            "visibility": self.visibility.value,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TestSpec":
        return cls(
            name=str(value["name"]),
            code=str(value["code"]),
            visibility=str(value.get("visibility", TestVisibility.HIDDEN.value)),
            kind=str(value.get("kind", "unit")),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Canonical representation of a coding task."""

    task_id: str
    source: str
    prompt: str
    split: Split | str
    entry_point: str | None = None
    starter_code: str = ""
    public_tests: tuple[TestSpec, ...] = field(default_factory=tuple)
    hidden_tests: tuple[TestSpec, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.split, str):
            object.__setattr__(self, "split", Split(self.split))
        object.__setattr__(
            self,
            "public_tests",
            tuple(_coerce_test(test, TestVisibility.PUBLIC) for test in self.public_tests),
        )
        object.__setattr__(
            self,
            "hidden_tests",
            tuple(_coerce_test(test, TestVisibility.HIDDEN) for test in self.hidden_tests),
        )
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))
        if isinstance(self.resource_limits, dict):
            object.__setattr__(
                self,
                "resource_limits",
                ResourceLimits.from_dict(self.resource_limits),
            )

        if not self.task_id.strip():
            raise ValueError("task_id must not be empty")
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")

    @property
    def all_tests(self) -> tuple[TestSpec, ...]:
        return self.public_tests + self.hidden_tests

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "prompt": self.prompt,
            "split": self.split.value,
            "entry_point": self.entry_point,
            "starter_code": self.starter_code,
            "public_tests": [test.to_dict() for test in self.public_tests],
            "hidden_tests": [test.to_dict() for test in self.hidden_tests],
            "tags": list(self.tags),
            "resource_limits": self.resource_limits.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(value["task_id"]),
            source=str(value["source"]),
            prompt=str(value["prompt"]),
            split=str(value["split"]),
            entry_point=value.get("entry_point"),
            starter_code=str(value.get("starter_code", "")),
            public_tests=tuple(TestSpec.from_dict(test) for test in value.get("public_tests", [])),
            hidden_tests=tuple(TestSpec.from_dict(test) for test in value.get("hidden_tests", [])),
            tags=tuple(str(tag) for tag in value.get("tags", [])),
            resource_limits=ResourceLimits.from_dict(value.get("resource_limits")),
            metadata={str(key): str(item) for key, item in value.get("metadata", {}).items()},
        )


def _coerce_test(test: TestSpec | dict[str, Any], expected_visibility: TestVisibility) -> TestSpec:
    test_spec = TestSpec.from_dict(test) if isinstance(test, dict) else test
    if test_spec.visibility != expected_visibility:
        test_spec = TestSpec(
            name=test_spec.name,
            code=test_spec.code,
            visibility=expected_visibility,
            kind=test_spec.kind,
        )
    return test_spec
