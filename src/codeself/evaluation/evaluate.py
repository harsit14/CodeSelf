"""Evaluation summaries for rollout JSONL files."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, median, pstdev
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
    pass_at: dict[str, float] = field(default_factory=dict)
    degenerate_outputs: int = 0
    response_token_counts: tuple[int, ...] = field(default_factory=tuple)

    @property
    def pass_at_1_observed(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.correct_samples / self.total_samples

    @property
    def mean_reward(self) -> float:
        return fmean(self.rewards) if self.rewards else 0.0

    @property
    def degenerate_output_rate(self) -> float:
        return _safe_rate(self.degenerate_outputs, self.total_samples)

    @property
    def response_token_mean(self) -> float:
        return fmean(self.response_token_counts) if self.response_token_counts else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_samples": self.total_samples,
            "correct_samples": self.correct_samples,
            "parse_failures": self.parse_failures,
            "pass_at_1_observed": self.pass_at_1_observed,
            "pass_at": dict(self.pass_at),
            "mean_reward": self.mean_reward,
            "degenerate_outputs": self.degenerate_outputs,
            "degenerate_output_rate": self.degenerate_output_rate,
            "response_token_mean": self.response_token_mean,
        }


@dataclass(frozen=True)
class GroupEvaluation:
    """Dashboard-ready metrics for a metadata-defined rollout group."""

    group_field: str
    group_value: str
    rollout_count: int
    task_count: int
    pass_at: dict[str, float]
    parse_failure_rate: float
    execution_pass_rate: float
    reward_mean: float
    reward_median: float
    reward_min: float
    reward_max: float
    degenerate_output_rate: float
    response_token_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_field": self.group_field,
            "group_value": self.group_value,
            "rollout_count": self.rollout_count,
            "task_count": self.task_count,
            "pass_at": dict(self.pass_at),
            "parse_failure_rate": self.parse_failure_rate,
            "execution_pass_rate": self.execution_pass_rate,
            "reward_mean": self.reward_mean,
            "reward_median": self.reward_median,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "degenerate_output_rate": self.degenerate_output_rate,
            "response_token_mean": self.response_token_mean,
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
    reward_std: float
    informative_prompt_fraction: float
    degenerate_output_rate: float
    response_token_mean: float
    response_token_median: float
    response_token_max: int
    task_summaries: tuple[TaskEvaluation, ...] = field(default_factory=tuple)
    group_summaries: tuple[GroupEvaluation, ...] = field(default_factory=tuple)

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
            "reward_std": self.reward_std,
            "informative_prompt_fraction": self.informative_prompt_fraction,
            "degenerate_output_rate": self.degenerate_output_rate,
            "response_token_mean": self.response_token_mean,
            "response_token_median": self.response_token_median,
            "response_token_max": self.response_token_max,
            "task_summaries": [task.to_dict() for task in self.task_summaries],
            "group_summaries": [group.to_dict() for group in self.group_summaries],
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
            f"- Reward std: {self.reward_std:.4f}",
            f"- Informative prompt fraction: {self.informative_prompt_fraction:.4f}",
            f"- Degenerate output rate: {self.degenerate_output_rate:.4f}",
            f"- Mean response tokens: {self.response_token_mean:.2f}",
            f"- Median response tokens: {self.response_token_median:.2f}",
            f"- Max response tokens: {self.response_token_max}",
            "",
            "## pass@k",
            "",
        ]
        for key, value in sorted(self.pass_at.items()):
            lines.append(f"- {key}: {value:.4f}")
        if self.group_summaries:
            lines.extend(
                [
                    "",
                    "## Group Summary",
                    "",
                    "| Field | Value | Rollouts | Tasks | pass@1 | Pass Rate | "
                    "Degenerate | Mean Reward | Mean Tokens |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for group in self.group_summaries:
                lines.append(
                    f"| {group.group_field} | {group.group_value} | "
                    f"{group.rollout_count} | {group.task_count} | "
                    f"{group.pass_at.get('pass@1', 0.0):.4f} | "
                    f"{group.execution_pass_rate:.4f} | "
                    f"{group.degenerate_output_rate:.4f} | "
                    f"{group.reward_mean:.4f} | {group.response_token_mean:.2f} |"
                )
        lines.extend(
            [
                "",
                "## Per-task Summary",
                "",
                "| Task | Samples | Correct | Parse Failures | pass@1 | "
                "Degenerate | Mean Reward | Mean Tokens |",
            ]
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for task in self.task_summaries:
            lines.append(
                f"| {task.task_id} | {task.total_samples} | {task.correct_samples} | "
                f"{task.parse_failures} | {task.pass_at.get('pass@1', 0.0):.4f} | "
                f"{task.degenerate_output_rate:.4f} | {task.mean_reward:.4f} | "
                f"{task.response_token_mean:.2f} |"
            )
        return "\n".join(lines) + "\n"


def evaluate_rollouts(
    records: list[RolloutRecord],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    max_samples_per_task: int | None = None,
    group_by: tuple[str, ...] = (),
) -> EvaluationSummary:
    """Build aggregate evaluation metrics from rollout records."""

    if max_samples_per_task is not None and max_samples_per_task <= 0:
        raise ValueError("max_samples_per_task must be positive when set")
    records = _limit_samples_per_task(records, max_samples_per_task)
    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)

    task_summaries = tuple(
        _task_summary(task_id, grouped[task_id], ks=ks) for task_id in sorted(grouped)
    )
    rewards = [float(record.reward.get("reward", 0.0)) for record in records]
    response_tokens = [_response_token_count(record) for record in records]
    parsed_failures = [record for record in records if record.parsed.status.value != "ok"]
    passed = [record for record in records if bool(record.execution.get("passed"))]
    degenerate = [record for record in records if _is_degenerate_output(record)]
    outcomes = [(task.total_samples, task.correct_samples) for task in task_summaries]
    pass_at = {f"pass@{k}": mean_pass_at_k(outcomes, k) for k in ks}
    informative_tasks = [
        task for task in task_summaries if 0 < task.correct_samples < task.total_samples
    ]
    group_summaries = _group_summaries(records, ks=ks, group_by=group_by)

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
        reward_std=pstdev(rewards) if len(rewards) > 1 else 0.0,
        informative_prompt_fraction=_safe_rate(len(informative_tasks), len(task_summaries)),
        degenerate_output_rate=_safe_rate(len(degenerate), len(records)),
        response_token_mean=fmean(response_tokens) if response_tokens else 0.0,
        response_token_median=median(response_tokens) if response_tokens else 0.0,
        response_token_max=max(response_tokens) if response_tokens else 0,
        task_summaries=task_summaries,
        group_summaries=group_summaries,
    )


def evaluate_rollout_file(
    path: str | Path,
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    max_samples_per_task: int | None = None,
    group_by: tuple[str, ...] = (),
) -> EvaluationSummary:
    return evaluate_rollouts(
        read_rollouts_jsonl(path),
        ks=ks,
        max_samples_per_task=max_samples_per_task,
        group_by=group_by,
    )


def write_evaluation_report(summary: EvaluationSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        path.write_text(summary.to_markdown(), encoding="utf-8")


def _task_summary(
    task_id: str,
    records: list[RolloutRecord],
    *,
    ks: tuple[int, ...],
) -> TaskEvaluation:
    correct_samples = sum(1 for record in records if bool(record.execution.get("passed")))
    total_samples = len(records)
    return TaskEvaluation(
        task_id=task_id,
        total_samples=total_samples,
        correct_samples=correct_samples,
        parse_failures=sum(1 for record in records if record.parsed.status.value != "ok"),
        rewards=tuple(float(record.reward.get("reward", 0.0)) for record in records),
        pass_at={
            f"pass@{k}": mean_pass_at_k([(total_samples, correct_samples)], k)
            for k in ks
        },
        degenerate_outputs=sum(1 for record in records if _is_degenerate_output(record)),
        response_token_counts=tuple(_response_token_count(record) for record in records),
    )


def _group_summaries(
    records: list[RolloutRecord],
    *,
    ks: tuple[int, ...],
    group_by: tuple[str, ...],
) -> tuple[GroupEvaluation, ...]:
    summaries: list[GroupEvaluation] = []
    for field_name in group_by:
        groups: dict[str, list[RolloutRecord]] = defaultdict(list)
        for record in records:
            groups[_group_value(record, field_name)].append(record)
        for value in sorted(groups):
            group_records = groups[value]
            task_summaries = tuple(
                _task_summary(task_id, task_records, ks=ks)
                for task_id, task_records in _records_by_task(group_records).items()
            )
            rewards = [float(record.reward.get("reward", 0.0)) for record in group_records]
            response_tokens = [_response_token_count(record) for record in group_records]
            outcomes = [
                (task.total_samples, task.correct_samples) for task in task_summaries
            ]
            summaries.append(
                GroupEvaluation(
                    group_field=field_name,
                    group_value=value,
                    rollout_count=len(group_records),
                    task_count=len(task_summaries),
                    pass_at={f"pass@{k}": mean_pass_at_k(outcomes, k) for k in ks},
                    parse_failure_rate=_safe_rate(
                        sum(
                            1
                            for record in group_records
                            if record.parsed.status.value != "ok"
                        ),
                        len(group_records),
                    ),
                    execution_pass_rate=_safe_rate(
                        sum(1 for record in group_records if record.execution.get("passed")),
                        len(group_records),
                    ),
                    reward_mean=fmean(rewards) if rewards else 0.0,
                    reward_median=median(rewards) if rewards else 0.0,
                    reward_min=min(rewards) if rewards else 0.0,
                    reward_max=max(rewards) if rewards else 0.0,
                    degenerate_output_rate=_safe_rate(
                        sum(1 for record in group_records if _is_degenerate_output(record)),
                        len(group_records),
                    ),
                    response_token_mean=fmean(response_tokens) if response_tokens else 0.0,
                )
            )
    return tuple(summaries)


def _records_by_task(records: list[RolloutRecord]) -> dict[str, list[RolloutRecord]]:
    grouped: dict[str, list[RolloutRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task_id].append(record)
    return {task_id: grouped[task_id] for task_id in sorted(grouped)}


def _limit_samples_per_task(
    records: list[RolloutRecord],
    max_samples_per_task: int | None,
) -> list[RolloutRecord]:
    if max_samples_per_task is None:
        return list(records)
    selected: list[RolloutRecord] = []
    for task_records in _records_by_task(records).values():
        selected.extend(
            sorted(task_records, key=lambda record: record.sample_index)[
                :max_samples_per_task
            ]
        )
    return selected


def _group_value(record: RolloutRecord, field_name: str) -> str:
    value = _lookup_record_field(record, field_name)
    if value is None:
        return "missing"
    return str(value)


def _lookup_record_field(record: RolloutRecord, field_name: str) -> object | None:
    value: object = record
    for part in field_name.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _response_token_count(record: RolloutRecord) -> int:
    value = record.metadata.get("response_tokens")
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    return len(record.raw_completion.split())


def _is_degenerate_output(record: RolloutRecord) -> bool:
    raw_text = record.raw_completion.strip()
    code = record.parsed.code.strip()
    if not raw_text or not code:
        return True
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) <= 0.5:
        return True
    tokens = raw_text.split()
    return len(tokens) >= 20 and len(set(tokens)) / len(tokens) <= 0.2


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
