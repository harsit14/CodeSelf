"""Paired statistical tests for final held-out evaluation."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeself.agent import RolloutRecord, read_rollouts_jsonl


@dataclass(frozen=True)
class PairedTaskOutcome:
    """Pass/fail outcome for one task under two systems."""

    task_id: str
    base_passed: bool
    candidate_passed: bool
    base_reward: float
    candidate_reward: float

    @property
    def delta(self) -> int:
        return int(self.candidate_passed) - int(self.base_passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "base_passed": self.base_passed,
            "candidate_passed": self.candidate_passed,
            "base_reward": self.base_reward,
            "candidate_reward": self.candidate_reward,
            "delta": self.delta,
        }


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar/binomial result over discordant pairs."""

    both_failed: int
    base_only: int
    candidate_only: int
    both_passed: int
    discordant: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "both_failed": self.both_failed,
            "base_only": self.base_only,
            "candidate_only": self.candidate_only,
            "both_passed": self.both_passed,
            "discordant": self.discordant,
            "p_value": self.p_value,
        }


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval."""

    mean: float
    lower: float
    upper: float
    confidence: float
    samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class ComparisonSummary:
    """Final paired comparison summary."""

    task_count: int
    sample_index: int
    base_pass_rate: float
    candidate_pass_rate: float
    delta_pass_rate: float
    delta_pp: float
    base_mean_reward: float
    candidate_mean_reward: float
    reward_delta: float
    mcnemar: McNemarResult
    bootstrap_delta: BootstrapCI
    outcomes: tuple[PairedTaskOutcome, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_count": self.task_count,
            "sample_index": self.sample_index,
            "base_pass_rate": self.base_pass_rate,
            "candidate_pass_rate": self.candidate_pass_rate,
            "delta_pass_rate": self.delta_pass_rate,
            "delta_pp": self.delta_pp,
            "base_mean_reward": self.base_mean_reward,
            "candidate_mean_reward": self.candidate_mean_reward,
            "reward_delta": self.reward_delta,
            "mcnemar": self.mcnemar.to_dict(),
            "bootstrap_delta": self.bootstrap_delta.to_dict(),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }

    def to_markdown(self) -> str:
        improved = [outcome.task_id for outcome in self.outcomes if outcome.delta > 0]
        regressed = [outcome.task_id for outcome in self.outcomes if outcome.delta < 0]
        lines = [
            "# Final Paired Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Tasks compared: {self.task_count}",
            f"- Sample index: {self.sample_index}",
            f"- Base pass@1: {self.base_pass_rate:.4f}",
            f"- Candidate pass@1: {self.candidate_pass_rate:.4f}",
            f"- Delta: {self.delta_pp:.2f} percentage points",
            f"- Base mean reward: {self.base_mean_reward:.4f}",
            f"- Candidate mean reward: {self.candidate_mean_reward:.4f}",
            f"- Reward delta: {self.reward_delta:.4f}",
            "",
            "## Paired Statistical Test",
            "",
            f"- Both failed: {self.mcnemar.both_failed}",
            f"- Base only passed: {self.mcnemar.base_only}",
            f"- Candidate only passed: {self.mcnemar.candidate_only}",
            f"- Both passed: {self.mcnemar.both_passed}",
            f"- Discordant pairs: {self.mcnemar.discordant}",
            f"- Exact McNemar p-value: {self.mcnemar.p_value:.6f}",
            "",
            "## Bootstrap CI",
            "",
            f"- Mean delta: {self.bootstrap_delta.mean * 100:.2f} percentage points",
            f"- {self.bootstrap_delta.confidence:.0%} CI: "
            f"[{self.bootstrap_delta.lower * 100:.2f}, {self.bootstrap_delta.upper * 100:.2f}] percentage points",
            f"- Bootstrap samples: {self.bootstrap_delta.samples}",
            "",
            "## Task Flips",
            "",
            f"- Improved tasks: {len(improved)}",
            f"- Regressed tasks: {len(regressed)}",
        ]
        if improved:
            lines.append(f"- Improved task IDs: {', '.join(improved)}")
        if regressed:
            lines.append(f"- Regressed task IDs: {', '.join(regressed)}")
        lines.extend(
            [
                "",
                "## Per-task Outcomes",
                "",
                "| Task | Base | Candidate | Base Reward | Candidate Reward | Delta |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for outcome in self.outcomes:
            lines.append(
                f"| {outcome.task_id} | {int(outcome.base_passed)} | "
                f"{int(outcome.candidate_passed)} | {outcome.base_reward:.4f} | "
                f"{outcome.candidate_reward:.4f} | {outcome.delta} |"
            )
        return "\n".join(lines) + "\n"


def compare_rollouts(
    base_records: list[RolloutRecord],
    candidate_records: list[RolloutRecord],
    *,
    sample_index: int = 0,
    bootstrap_samples: int = 2000,
    seed: int = 20260601,
    confidence: float = 0.95,
) -> ComparisonSummary:
    """Compare base and candidate rollout records on common task IDs."""

    outcomes = paired_task_outcomes(
        base_records,
        candidate_records,
        sample_index=sample_index,
    )
    if not outcomes:
        raise ValueError("no common task outcomes to compare")

    base_pass_rate = _mean(int(outcome.base_passed) for outcome in outcomes)
    candidate_pass_rate = _mean(int(outcome.candidate_passed) for outcome in outcomes)
    base_reward = _mean(outcome.base_reward for outcome in outcomes)
    candidate_reward = _mean(outcome.candidate_reward for outcome in outcomes)
    delta = candidate_pass_rate - base_pass_rate
    return ComparisonSummary(
        task_count=len(outcomes),
        sample_index=sample_index,
        base_pass_rate=base_pass_rate,
        candidate_pass_rate=candidate_pass_rate,
        delta_pass_rate=delta,
        delta_pp=delta * 100,
        base_mean_reward=base_reward,
        candidate_mean_reward=candidate_reward,
        reward_delta=candidate_reward - base_reward,
        mcnemar=exact_mcnemar(outcomes),
        bootstrap_delta=bootstrap_delta_ci(
            outcomes,
            samples=bootstrap_samples,
            seed=seed,
            confidence=confidence,
        ),
        outcomes=tuple(sorted(outcomes, key=lambda outcome: outcome.task_id)),
    )


def compare_rollout_files(
    base_path: str | Path,
    candidate_path: str | Path,
    **kwargs: Any,
) -> ComparisonSummary:
    return compare_rollouts(read_rollouts_jsonl(base_path), read_rollouts_jsonl(candidate_path), **kwargs)


def paired_task_outcomes(
    base_records: list[RolloutRecord],
    candidate_records: list[RolloutRecord],
    *,
    sample_index: int = 0,
) -> list[PairedTaskOutcome]:
    base = _record_by_task(base_records, sample_index)
    candidate = _record_by_task(candidate_records, sample_index)
    common_task_ids = sorted(set(base) & set(candidate))
    return [
        PairedTaskOutcome(
            task_id=task_id,
            base_passed=bool(base[task_id].execution.get("passed")),
            candidate_passed=bool(candidate[task_id].execution.get("passed")),
            base_reward=float(base[task_id].reward.get("reward", 0.0)),
            candidate_reward=float(candidate[task_id].reward.get("reward", 0.0)),
        )
        for task_id in common_task_ids
    ]


def exact_mcnemar(outcomes: list[PairedTaskOutcome]) -> McNemarResult:
    """Exact two-sided McNemar test via binomial discordant-pair test."""

    both_failed = sum(
        1 for outcome in outcomes if not outcome.base_passed and not outcome.candidate_passed
    )
    base_only = sum(1 for outcome in outcomes if outcome.base_passed and not outcome.candidate_passed)
    candidate_only = sum(
        1 for outcome in outcomes if not outcome.base_passed and outcome.candidate_passed
    )
    both_passed = sum(1 for outcome in outcomes if outcome.base_passed and outcome.candidate_passed)
    discordant = base_only + candidate_only
    p_value = _exact_binomial_two_sided(min(base_only, candidate_only), discordant)
    return McNemarResult(
        both_failed=both_failed,
        base_only=base_only,
        candidate_only=candidate_only,
        both_passed=both_passed,
        discordant=discordant,
        p_value=p_value,
    )


def bootstrap_delta_ci(
    outcomes: list[PairedTaskOutcome],
    *,
    samples: int = 2000,
    seed: int = 20260601,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Bootstrap paired pass-rate deltas over tasks."""

    if not outcomes:
        raise ValueError("outcomes must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        resampled = [outcomes[rng.randrange(len(outcomes))] for _ in outcomes]
        deltas.append(_mean(outcome.delta for outcome in resampled))
    deltas.sort()
    alpha = (1 - confidence) / 2
    lower = deltas[_quantile_index(alpha, len(deltas))]
    upper = deltas[_quantile_index(1 - alpha, len(deltas))]
    return BootstrapCI(
        mean=_mean(deltas),
        lower=lower,
        upper=upper,
        confidence=confidence,
        samples=samples,
    )


def write_comparison_report(summary: ComparisonSummary, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(summary.to_markdown(), encoding="utf-8")


def _record_by_task(records: list[RolloutRecord], sample_index: int) -> dict[str, RolloutRecord]:
    grouped: dict[str, list[RolloutRecord]] = {}
    for record in records:
        grouped.setdefault(record.task_id, []).append(record)

    selected: dict[str, RolloutRecord] = {}
    for task_id, task_records in grouped.items():
        sorted_records = sorted(task_records, key=lambda record: record.sample_index)
        exact = [record for record in sorted_records if record.sample_index == sample_index]
        selected[task_id] = exact[0] if exact else sorted_records[0]
    return selected


def _exact_binomial_two_sided(smaller_count: int, total: int) -> float:
    if total == 0:
        return 1.0
    cumulative = sum(math.comb(total, i) for i in range(smaller_count + 1)) / (2**total)
    return min(1.0, 2 * cumulative)


def _quantile_index(probability: float, length: int) -> int:
    return max(0, min(length - 1, int(probability * (length - 1))))


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
