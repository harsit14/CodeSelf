"""Multi-cycle GRPO online training orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from codeself.agent.generation import CodeGenerator
from codeself.agent.prompts import PromptTemplate
from codeself.datasets import TaskSpec
from codeself.execution import SandboxedTestRunner
from codeself.rewards import RewardScorer
from codeself.training.common import TokenizerEngine
from codeself.training.grpo_cycle import (
    GRPORolloutTrainingCycleConfig,
    GRPORolloutTrainingCycleResult,
    run_grpo_rollout_training_cycle,
)
from codeself.training.grpo_torch import require_torch


@dataclass(frozen=True)
class GRPOOnlineTrainingConfig:
    """Config for repeated rollout-training cycles."""

    cycles: int = 1
    cycle: GRPORolloutTrainingCycleConfig = field(
        default_factory=GRPORolloutTrainingCycleConfig
    )
    seed_stride: int = 1
    sync_old_policy_before_cycle: bool = True

    def __post_init__(self) -> None:
        if self.cycles <= 0:
            raise ValueError("cycles must be positive")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")

    def cycle_config_for(self, cycle: int) -> GRPORolloutTrainingCycleConfig:
        if cycle <= 0 or cycle > self.cycles:
            raise ValueError("cycle must be in [1, cycles]")
        return replace(
            self.cycle,
            seed=self.cycle.seed + (cycle - 1) * self.seed_stride,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cycles": self.cycles,
            "cycle": self.cycle.to_dict(),
            "seed_stride": self.seed_stride,
            "sync_old_policy_before_cycle": self.sync_old_policy_before_cycle,
        }


@dataclass(frozen=True)
class GRPOOnlineTrainingStep:
    """One completed online GRPO cycle."""

    cycle: int
    seed: int
    result: GRPORolloutTrainingCycleResult
    old_policy_synced: bool = False

    @property
    def rollout_count(self) -> int:
        return self.result.rollout_count

    @property
    def records_used(self) -> int:
        return self.result.rollout_batch.records_used

    @property
    def mean_reward(self) -> float:
        return self.result.rollout_batch.batch.mean_reward

    @property
    def optimizer_step_count(self) -> int:
        return self.result.training.loop.optimizer_step_count

    def to_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "seed": self.seed,
            "old_policy_synced": self.old_policy_synced,
            "rollout_count": self.rollout_count,
            "records_used": self.records_used,
            "mean_reward": self.mean_reward,
            "optimizer_step_count": self.optimizer_step_count,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class GRPOOnlineTrainingResult:
    """Result summary for repeated GRPO online training cycles."""

    config: GRPOOnlineTrainingConfig
    steps: tuple[GRPOOnlineTrainingStep, ...]
    artifact_dir: str | None = None

    @property
    def cycle_count(self) -> int:
        return len(self.steps)

    @property
    def total_rollouts(self) -> int:
        return sum(step.rollout_count for step in self.steps)

    @property
    def records_used(self) -> int:
        return sum(step.records_used for step in self.steps)

    @property
    def total_optimizer_steps(self) -> int:
        return sum(step.optimizer_step_count for step in self.steps)

    @property
    def mean_reward(self) -> float:
        if self.records_used == 0:
            return 0.0
        weighted_reward = sum(step.mean_reward * step.records_used for step in self.steps)
        return weighted_reward / self.records_used

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "cycle_count": self.cycle_count,
            "total_rollouts": self.total_rollouts,
            "records_used": self.records_used,
            "total_optimizer_steps": self.total_optimizer_steps,
            "mean_reward": self.mean_reward,
            "artifact_dir": self.artifact_dir,
            "steps": [step.to_dict() for step in self.steps],
        }


def run_grpo_online_training(
    tasks: Sequence[TaskSpec],
    *,
    generator: CodeGenerator,
    prompt_template: PromptTemplate,
    tokenizer: TokenizerEngine,
    policy_model: Any,
    old_policy_model: Any | None = None,
    reference_model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scorer: RewardScorer | None = None,
    runner: SandboxedTestRunner | None = None,
    config: GRPOOnlineTrainingConfig | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    artifact_dir: str | Path | None = None,
) -> GRPOOnlineTrainingResult:
    """Run repeated collect -> execute -> score -> optimize GRPO cycles."""

    task_list = list(tasks)
    if not task_list:
        raise ValueError("GRPO online training requires at least one task")

    online_config = config or GRPOOnlineTrainingConfig()
    output_dir = Path(artifact_dir) if artifact_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    active_optimizer = optimizer or _build_optimizer(policy_model, online_config)
    steps: list[GRPOOnlineTrainingStep] = []
    for cycle in range(1, online_config.cycles + 1):
        cycle_config = online_config.cycle_config_for(cycle)
        cycle_dir = output_dir / f"cycle_{cycle:04d}" if output_dir is not None else None
        if cycle_dir is not None:
            cycle_dir.mkdir(parents=True, exist_ok=True)
        old_policy_synced = False
        if old_policy_model is not None and online_config.sync_old_policy_before_cycle:
            _sync_model_state(policy_model, old_policy_model)
            old_policy_synced = True
        result = run_grpo_rollout_training_cycle(
            task_list,
            generator=generator,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            policy_model=policy_model,
            old_policy_model=old_policy_model,
            reference_model=reference_model,
            optimizer=active_optimizer,
            scheduler=scheduler,
            scorer=scorer,
            runner=runner,
            config=cycle_config,
            metadata={
                "online_cycle": cycle,
                "online_total_cycles": online_config.cycles,
                **(metadata or {}),
            },
            rollouts_path=cycle_dir / "rollouts.jsonl" if cycle_dir is not None else None,
            metrics_path=cycle_dir / "metrics.jsonl" if cycle_dir is not None else None,
            checkpoint_path=cycle_dir / "checkpoint.json" if cycle_dir is not None else None,
            state_dir=cycle_dir / "state" if cycle_dir is not None else None,
        )
        steps.append(
            GRPOOnlineTrainingStep(
                cycle=cycle,
                seed=cycle_config.seed,
                result=result,
                old_policy_synced=old_policy_synced,
            )
        )

    return GRPOOnlineTrainingResult(
        config=online_config,
        steps=tuple(steps),
        artifact_dir=str(output_dir) if output_dir is not None else None,
    )


def _build_optimizer(policy_model: Any, config: GRPOOnlineTrainingConfig) -> Any:
    torch = require_torch()
    parameters = _trainable_parameters(policy_model)
    if not parameters:
        raise ValueError("policy_model must expose at least one trainable parameter")
    optimizer_config = config.cycle.training.optimizer
    return torch.optim.AdamW(
        parameters,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )


def _sync_model_state(source_model: Any, target_model: Any) -> None:
    if not hasattr(source_model, "state_dict"):
        raise TypeError("policy_model must define state_dict() to sync old-policy state")
    if not hasattr(target_model, "load_state_dict"):
        raise TypeError("old_policy_model must define load_state_dict()")
    target_model.load_state_dict(source_model.state_dict())


def _trainable_parameters(model: Any) -> list[Any]:
    if not hasattr(model, "parameters"):
        raise TypeError("policy_model must define parameters()")
    return [
        parameter
        for parameter in model.parameters()
        if bool(getattr(parameter, "requires_grad", True))
    ]
