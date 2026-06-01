"""Compare GRPO and PPO smoke diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlgorithmComparison:
    """Comparison between final GRPO and PPO smoke metrics."""

    grpo: dict[str, Any]
    ppo: dict[str, Any]
    rollout_budget_matched: bool
    mean_reward_delta_grpo_minus_ppo: float
    pass_rate_delta_grpo_minus_ppo: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "grpo": dict(self.grpo),
            "ppo": dict(self.ppo),
            "rollout_budget_matched": self.rollout_budget_matched,
            "mean_reward_delta_grpo_minus_ppo": self.mean_reward_delta_grpo_minus_ppo,
            "pass_rate_delta_grpo_minus_ppo": self.pass_rate_delta_grpo_minus_ppo,
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        return (
            "# PPO versus GRPO Smoke Comparison\n\n"
            "| Algorithm | Rollouts | Mean Reward | Pass Rate | Parse Failure | Key Signal |\n"
            "|---|---:|---:|---:|---:|---:|\n"
            f"| GRPO | {self.grpo.get('rollout_count', 0)} | "
            f"{float(self.grpo.get('mean_reward', 0.0)):.4f} | "
            f"{float(self.grpo.get('execution_pass_rate', 0.0)):.4f} | "
            f"{float(self.grpo.get('parse_failure_rate', 0.0)):.4f} | "
            f"{float(self.grpo.get('informative_prompt_fraction', 0.0)):.4f} informative |\n"
            f"| PPO | {self.ppo.get('rollout_count', 0)} | "
            f"{float(self.ppo.get('mean_reward', 0.0)):.4f} | "
            f"{float(self.ppo.get('execution_pass_rate', 0.0)):.4f} | "
            f"{float(self.ppo.get('parse_failure_rate', 0.0)):.4f} | "
            f"{float(self.ppo.get('clipped_fraction', 0.0)):.4f} clipped |\n\n"
            "## Summary\n\n"
            f"- Rollout budget matched: {self.rollout_budget_matched}\n"
            f"- Mean reward delta, GRPO minus PPO: {self.mean_reward_delta_grpo_minus_ppo:.4f}\n"
            f"- Pass-rate delta, GRPO minus PPO: {self.pass_rate_delta_grpo_minus_ppo:.4f}\n"
            f"- Notes: {self.notes}\n"
        )


def compare_algorithm_metrics(grpo_metrics: dict[str, Any], ppo_metrics: dict[str, Any]) -> AlgorithmComparison:
    grpo_rollouts = int(grpo_metrics.get("rollout_count", 0))
    ppo_rollouts = int(ppo_metrics.get("rollout_count", 0))
    return AlgorithmComparison(
        grpo=dict(grpo_metrics),
        ppo=dict(ppo_metrics),
        rollout_budget_matched=grpo_rollouts == ppo_rollouts,
        mean_reward_delta_grpo_minus_ppo=float(grpo_metrics.get("mean_reward", 0.0))
        - float(ppo_metrics.get("mean_reward", 0.0)),
        pass_rate_delta_grpo_minus_ppo=float(grpo_metrics.get("execution_pass_rate", 0.0))
        - float(ppo_metrics.get("execution_pass_rate", 0.0)),
        notes=(
            "Smoke comparison checks metric/report plumbing only. Real PPO and GRPO "
            "require matched training budgets, logprobs, and model checkpoints."
        ),
    )


def read_last_metric(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    last: dict[str, Any] | None = None
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last = json.loads(stripped)
    if last is None:
        raise ValueError(f"no metrics found in {input_path}")
    return last


def write_algorithm_comparison(comparison: AlgorithmComparison, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        output_path.write_text(
            json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        output_path.write_text(comparison.to_markdown(), encoding="utf-8")
