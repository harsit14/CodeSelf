"""Runtime-efficiency diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from codeself.execution import PhaseStatus, TaskRunResult


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Efficiency metrics for a task run."""

    duration_seconds: float
    timeout: bool
    reference_duration_seconds: float | None
    relative_runtime_score: float | None

    def to_dict(self) -> dict[str, float | int | bool]:
        values: dict[str, float | int | bool] = {
            "duration_seconds": self.duration_seconds,
            "timeout": self.timeout,
        }
        if self.reference_duration_seconds is not None:
            values["reference_duration_seconds"] = self.reference_duration_seconds
        if self.relative_runtime_score is not None:
            values["relative_runtime_score"] = self.relative_runtime_score
        return values


def measure_efficiency(
    result: TaskRunResult,
    *,
    reference_duration_seconds: float | None = None,
) -> EfficiencyMetrics:
    """Measure runtime diagnostics without making them part of the MVP reward."""

    timeout = any(phase.status == PhaseStatus.TIMEOUT for phase in result.phases)
    relative_score = None
    if result.passed and reference_duration_seconds and result.duration_seconds > 0:
        relative_score = max(
            -1.0,
            min(1.0, math.log(reference_duration_seconds / result.duration_seconds)),
        )
    return EfficiencyMetrics(
        duration_seconds=result.duration_seconds,
        timeout=timeout,
        reference_duration_seconds=reference_duration_seconds,
        relative_runtime_score=relative_score,
    )
