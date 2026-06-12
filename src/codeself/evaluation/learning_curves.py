"""Learning-curve extraction from online training artifact directories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from codeself.agent import read_rollouts_jsonl
from codeself.evaluation.evaluate import evaluate_rollouts


@dataclass(frozen=True)
class LearningCurvePoint:
    """One online training cycle summarized for plotting."""

    cycle: int
    rollout_count: int
    train_pass_at_1: float
    train_reward_mean: float
    train_degenerate_rate: float
    train_response_token_mean: float
    optimizer_steps: int
    train_loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    kl_loss: float
    mean_approx_kl: float
    eval_pass_at_1: float | None = None
    eval_reward_mean: float | None = None
    eval_degenerate_rate: float | None = None
    eval_response_token_mean: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "rollout_count": self.rollout_count,
            "train_pass_at_1": self.train_pass_at_1,
            "train_reward_mean": self.train_reward_mean,
            "train_degenerate_rate": self.train_degenerate_rate,
            "train_response_token_mean": self.train_response_token_mean,
            "optimizer_steps": self.optimizer_steps,
            "train_loss": self.train_loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy_loss": self.entropy_loss,
            "kl_loss": self.kl_loss,
            "mean_approx_kl": self.mean_approx_kl,
            "eval_pass_at_1": self.eval_pass_at_1,
            "eval_reward_mean": self.eval_reward_mean,
            "eval_degenerate_rate": self.eval_degenerate_rate,
            "eval_response_token_mean": self.eval_response_token_mean,
        }


@dataclass(frozen=True)
class LearningCurveRun:
    """A labeled online training run summarized over cycles."""

    label: str
    artifact_dir: str
    points: tuple[LearningCurvePoint, ...]

    @property
    def cycle_count(self) -> int:
        return len(self.points)

    @property
    def total_rollouts(self) -> int:
        return sum(point.rollout_count for point in self.points)

    @property
    def total_optimizer_steps(self) -> int:
        return sum(point.optimizer_steps for point in self.points)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "artifact_dir": self.artifact_dir,
            "cycle_count": self.cycle_count,
            "total_rollouts": self.total_rollouts,
            "total_optimizer_steps": self.total_optimizer_steps,
            "points": [point.to_dict() for point in self.points],
        }


def collect_learning_curve(label: str, artifact_dir: str | Path) -> LearningCurveRun:
    """Collect one online training artifact directory into curve points."""

    root = Path(artifact_dir)
    if not root.exists():
        raise ValueError(f"artifact_dir does not exist: {root}")
    cycle_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("cycle_")
    ]
    if not cycle_dirs:
        raise ValueError(f"artifact_dir contains no cycle_* directories: {root}")
    points = tuple(
        _cycle_point(path, fallback_cycle=index + 1)
        for index, path in enumerate(cycle_dirs)
    )
    return LearningCurveRun(label=label, artifact_dir=str(root), points=points)


def collect_learning_curves(
    artifacts: dict[str, str | Path],
) -> dict[str, dict[str, object]]:
    """Collect many labeled artifact directories for dashboard rendering."""

    return {
        label: collect_learning_curve(label, path).to_dict()
        for label, path in artifacts.items()
    }


def _cycle_point(cycle_dir: Path, *, fallback_cycle: int) -> LearningCurvePoint:
    cycle = _cycle_number(cycle_dir.name, fallback_cycle)
    train_summary = _training_rollout_summary(cycle_dir / "rollouts.jsonl")
    metric_rows = _read_jsonl(cycle_dir / "metrics.jsonl")
    checkpoint = _read_json(cycle_dir / "checkpoint.json")
    evaluation = _read_json(cycle_dir / "evaluation.json")
    return LearningCurvePoint(
        cycle=cycle,
        rollout_count=int(train_summary.get("rollout_count", 0)),
        train_pass_at_1=_float(train_summary, "pass_at.pass@1"),
        train_reward_mean=float(train_summary.get("reward_mean", 0.0)),
        train_degenerate_rate=float(train_summary.get("degenerate_output_rate", 0.0)),
        train_response_token_mean=float(train_summary.get("response_token_mean", 0.0)),
        optimizer_steps=_optimizer_steps(metric_rows, checkpoint),
        train_loss=_mean_metric(metric_rows, "loss"),
        policy_loss=_mean_metric(metric_rows, "policy_loss"),
        value_loss=_mean_metric(metric_rows, "value_loss"),
        entropy_loss=_mean_metric(metric_rows, "entropy_loss"),
        kl_loss=_mean_metric(metric_rows, "kl_loss"),
        mean_approx_kl=_mean_metric(metric_rows, "mean_approx_kl"),
        eval_pass_at_1=_optional_float(evaluation, "pass_at.pass@1"),
        eval_reward_mean=_optional_float(evaluation, "reward_mean"),
        eval_degenerate_rate=_optional_float(evaluation, "degenerate_output_rate"),
        eval_response_token_mean=_optional_float(evaluation, "response_token_mean"),
    )


def _training_rollout_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return evaluate_rollouts(read_rollouts_jsonl(path), ks=(1,)).to_dict()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        value = json.loads(stripped)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _optimizer_steps(rows: list[dict[str, Any]], checkpoint: dict[str, Any]) -> int:
    count = sum(1 for row in rows if bool(_lookup(row, "metrics.optimizer_step")))
    if count:
        return count
    value = _lookup(checkpoint, "loop.optimizer_step_count")
    return int(value) if isinstance(value, (int, float)) else 0


def _mean_metric(rows: list[dict[str, Any]], metric_name: str) -> float:
    values = []
    for row in rows:
        value = _lookup(row, f"metrics.metrics.{metric_name}")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return fmean(values) if values else 0.0


def _cycle_number(name: str, fallback: int) -> int:
    try:
        return int(name.rsplit("_", 1)[-1])
    except ValueError:
        return fallback


def _optional_float(payload: dict[str, Any], path: str) -> float | None:
    if not payload:
        return None
    value = _lookup(payload, path)
    return float(value) if isinstance(value, (int, float)) else None


def _float(payload: dict[str, Any], path: str) -> float:
    value = _lookup(payload, path)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _lookup(payload: dict[str, Any], path: str) -> object | None:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
