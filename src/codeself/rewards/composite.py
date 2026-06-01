"""Composite reward scorer."""

from __future__ import annotations

from codeself.execution import TaskRunResult
from codeself.rewards.correctness import RewardBreakdown, RewardConfig, score_correctness
from codeself.rewards.efficiency import EfficiencyMetrics, measure_efficiency
from codeself.rewards.quality import QualityMetrics, measure_quality


class CompositeRewardScorer:
    """Scores a task run and attaches diagnostic metrics."""

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def score(
        self,
        result: TaskRunResult,
        *,
        solution_code: str = "",
        reference_duration_seconds: float | None = None,
    ) -> RewardBreakdown:
        breakdown = score_correctness(result, self.config)
        quality = measure_quality(solution_code) if solution_code else QualityMetrics.empty()
        efficiency = measure_efficiency(result, reference_duration_seconds=reference_duration_seconds)
        metrics = {
            **breakdown.metrics,
            **_prefixed("quality", quality.to_dict()),
            **_prefixed("efficiency", efficiency.to_dict()),
        }
        return RewardBreakdown(
            task_id=breakdown.task_id,
            reward_name=breakdown.reward_name,
            reward=breakdown.reward,
            score_before_penalty=breakdown.score_before_penalty,
            total_penalty=breakdown.total_penalty,
            components=breakdown.components,
            penalties=breakdown.penalties,
            metrics=metrics,
        )


def _prefixed(prefix: str, values: dict[str, float | int | bool]) -> dict[str, float | int | bool]:
    return {f"{prefix}_{key}": value for key, value in values.items()}
