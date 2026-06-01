"""pass@k estimators for code generation."""

from __future__ import annotations

from math import comb


def estimate_pass_at_k(total_samples: int, correct_samples: int, k: int) -> float:
    """Estimate pass@k with the HumanEval unbiased estimator.

    Args:
        total_samples: Number of generated samples for a task.
        correct_samples: Number of passing samples for the task.
        k: Number of samples considered for pass@k.
    """

    if total_samples < 0 or correct_samples < 0 or k <= 0:
        raise ValueError("total_samples/correct_samples must be non-negative and k must be positive")
    if correct_samples > total_samples:
        raise ValueError("correct_samples cannot exceed total_samples")
    if total_samples == 0:
        return 0.0
    if total_samples - correct_samples < k:
        return 1.0
    return 1.0 - comb(total_samples - correct_samples, k) / comb(total_samples, k)


def mean_pass_at_k(task_outcomes: list[tuple[int, int]], k: int) -> float:
    """Mean pass@k over tasks.

    Each tuple is `(total_samples, correct_samples)`.
    """

    if not task_outcomes:
        return 0.0
    return sum(estimate_pass_at_k(total, correct, k) for total, correct in task_outcomes) / len(
        task_outcomes
    )
