"""Evaluation summaries for rollout JSONL files."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median
from typing import Any

from codeself.agent import RolloutRecord, read_rollouts_jsonl
from codeself.evaluation.pass_at_k import mean_pass_at_k


@dataclass(frozen=True)
class TaskEvaluation:
    """Per-task evaluation summary."""

    task_id: str
    total_samples: int
    correct_samples: int
    parse_failures: int
    rewards: tuple[float, ...]

    @property
    def pass_at_1_observed(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "parse_failures": self.parse_failures,
            "pass_at_1_observed": self.pass_at_1_observed,
            "mean_reward": fmean(self.rewards) if self.rewards else 0.0,
        }


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation summary."""

    rollout_count: int
    task_count: int
    pass_at: dict[str, float]
    parse_failure_rate: float
    execution_pass_rate: float
    reward_mean: float
    reward_median: float
    reward_min: float
    reward_max: float
    informative_prompt_fraction: float
    task_summaries: tuple[TaskEvaluation, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_count": self.rollout_count,
            "task_count": self.task_count,
            "pass_at": dict(self.pass_at),
            "parse_failure_rate": self.parse_failure_rate,
            "execution_pass_rate": self.execution_pass_rate,
            "reward_mean": self.reward_mean,
            "reward_median": self.reward_median,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "informative_prompt_fraction": self.informative_prompt_fraction,
            "task_summaries": [task.to_dict() for task in self.task_summaries],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Baseline Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Rollouts: {self.rollout_count}",
            f"- Tasks: {self.task_count}",
            f"- Execution pass rate: {self.execution_pass_rate:.4f}",
            f"- Parser failure rate: {self.parse_failure_rate:.4f}",
            f"- Mean reward: {self.reward_mean:.4f}",
            f"- Median reward: {self.reward_median:.4f}",
            f"- Reward range: {self.reward_min:.4f} to {self.reward_max:.4f}",
            f"- Informative prompt fraction: {self.informative_prompt_fraction:.4f}",
            "",
            "## pass@k",
            "",
        ]
        for key, value in sorted(self.pass_at.items()):
            lines.append(f"- {key}: {value:.4f}")
        lines.extend(["", "## Per-task Summary", "", "| Task | Samples | Correct | Parse Failures | Mean Reward |"])
        lines.append("|---|---:|---:|---:|---:|")
        for task in self.task_summaries:
            lines.append(
                f"| {task.task_id} | {task.total_samples} | {task.correct_samples} | "
                f"{task.parse_failures} | {fmean(task.rewards) if task.rewards else 0.0:.4f} |"
            )
        return "\n".join(lines) + "\n"


def evaluate_rollouts(records: list[RolloutRecord], *, ks: tuple[int, ...] = (1, 5, 10)) -> EvaluationSummary:
    """Build aggregate evaluation metrics from rollout records."""

    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)

    task_summaries = tuple(
        _task_summary(task_id, grouped[task_id]) for task_id in sorted(grouped)
    )
    rewards = [float(record.reward.get("reward", 0.0)) for record in records]
    parsed_failures = [record for record in records if record.parsed.status.value != "ok"]
    passed = [record for record in records if bool(record.execution.get("passed"))]
    outcomes = [(task.total_samples, task.correct_samples) for task in task_summaries]
    pass_at = {f"pass@{k}": mean_pass_at_k(outcomes, k) for k in ks}
    informative_tasks = [
        task for task in task_summaries if 0 < task.correct_samples < task.total_samples
    ]

    return EvaluationSummary(
        rollout_count=len(records),
        task_count=len(task_summaries),
        pass_at=pass_at,
        parse_failure_rate=_safe_rate(len(parsed_failures), len(records)),
        execution_pass_rate=_safe_rate(len(passed), len(records)),
        reward_mean=fmean(rewards) if rewards else 0.0,
        reward_median=median(rewards) if rewards else 0.0,
        reward_min=min(rewards) if rewards else 0.0,
        reward_max=max(rewards) if rewards else 0.0,
        informative_prompt_fraction=_safe_rate(len(informative_tasks), len(task_summaries)),
        task_summaries=task_summaries,
    )


def evaluate_rollout_file(
    path: str | Path,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> EvaluationSummary:
    return evaluate_rollouts(read_rollouts_jsonl(path), ks=ks)


def write_evaluation_report(summary: EvaluationSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(summary.to_markdown(), encoding="utf-8")


def _task_summary(task_id: str, records: list[RolloutRecord]) -> TaskEvaluation:
    return TaskEvaluation(
        task_id=task_id,
        total_samples=len(records),
        correct_samples=sum(1 for record in records if bool(record.execution.get("passed"))),
        parse_failures=sum(1 for record in records if record.parsed.status.value != "ok"),
        rewards=tuple(float(record.reward.get("reward", 0.0)) for record in records),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
