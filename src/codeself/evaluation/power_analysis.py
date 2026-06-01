"""Simple power-analysis helpers for baseline planning."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Approximate detectable-effect result for paired pass/fail evals."""

    task_count: int
    baseline_pass_rate: float
    alpha: float
    z_alpha_two_sided: float
    minimum_detectable_effect_pp: float
    notes: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "task_count": self.task_count,
            "baseline_pass_rate": self.baseline_pass_rate,
            "alpha": self.alpha,
            "z_alpha_two_sided": self.z_alpha_two_sided,
            "minimum_detectable_effect_pp": self.minimum_detectable_effect_pp,
            "notes": self.notes,
        }


def approximate_minimum_detectable_effect(
    *,
    task_count: int,
    baseline_pass_rate: float,
    alpha: float = 0.05,
    z_alpha_two_sided: float = 1.96,
) -> PowerAnalysisResult:
    """Approximate a two-sided minimum detectable pass@1 delta in percentage points.

    This is intentionally conservative and simple. It uses a normal approximation
    to the standard error of a binomial pass rate, then doubles it to represent a
    clearly visible before/after separation. Final claims should still use paired
    tests on actual task outcomes.
    """

    if task_count <= 0:
        raise ValueError("task_count must be positive")
    if not 0 <= baseline_pass_rate <= 1:
        raise ValueError("baseline_pass_rate must be in [0, 1]")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")

    standard_error = math.sqrt(baseline_pass_rate * (1 - baseline_pass_rate) / task_count)
    minimum_detectable_effect = 2 * z_alpha_two_sided * standard_error
    return PowerAnalysisResult(
        task_count=task_count,
        baseline_pass_rate=baseline_pass_rate,
        alpha=alpha,
        z_alpha_two_sided=z_alpha_two_sided,
        minimum_detectable_effect_pp=minimum_detectable_effect * 100,
        notes=(
            "Normal-approximation planning estimate. Use paired tests and bootstrap "
            "confidence intervals for final claims."
        ),
    )
