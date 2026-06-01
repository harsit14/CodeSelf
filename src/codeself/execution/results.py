"""Execution result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PhaseStatus(str, Enum):
    """Status for one execution phase."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    SECURITY_REJECTED = "security_rejected"
    PARSE_ERROR = "parse_error"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True)
class SecurityFinding:
    """One static security finding."""

    rule: str
    message: str
    line: int | None = None
    severity: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class PhaseResult:
    """Result from one phase, such as syntax, import, public tests, or hidden tests."""

    name: str
    status: PhaseStatus
    duration_seconds: float = 0.0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    tests_run: int = 0

    @property
    def passed(self) -> bool:
        return self.status == PhaseStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error": self.error,
            "tests_run": self.tests_run,
        }


@dataclass(frozen=True)
class TaskRunResult:
    """End-to-end result for one generated solution on one task."""

    task_id: str
    status: PhaseStatus
    phases: tuple[PhaseResult, ...]
    security_findings: tuple[SecurityFinding, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return self.status == PhaseStatus.PASSED

    @property
    def duration_seconds(self) -> float:
        return sum(phase.duration_seconds for phase in self.phases)

    def phase(self, name: str) -> PhaseResult | None:
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "phases": [phase.to_dict() for phase in self.phases],
            "security_findings": [finding.to_dict() for finding in self.security_findings],
        }
