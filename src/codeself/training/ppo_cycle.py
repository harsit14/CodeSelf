"""One-cycle PPO orchestration from rollouts to model updates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from codeself.agent.generation import CodeGenerator
from codeself.agent.prompts import PromptTemplate
from codeself.agent.rollouts import RolloutRecord, generate_rollouts, write_rollouts_jsonl
from codeself.datasets import TaskSpec
from codeself.execution import SandboxedTestRunner
from codeself.rewards import RewardScorer
from codeself.training.common import RolloutRuntimeConfig, TokenizerEngine
from codeself.training.ppo_loss import PPOLossConfig
from codeself.training.ppo_rollouts import (
    PPORolloutBatchConfig,
    PPORolloutBatchResult,
    build_ppo_training_batch_from_rollouts,
)
from codeself.training.ppo_trainer import (
    PPOModelTrainingConfig,
    PPOModelTrainingResult,
    PPOValueEstimateProvider,
    run_ppo_model_training,
)


def _default_training_config() -> PPOModelTrainingConfig:
    return PPOModelTrainingConfig(loss=PPOLossConfig(kl_beta=0.0))


@dataclass(frozen=True)
class PPORolloutTrainingCycleConfig:
    """Config for one collect -> batch -> optimize PPO cycle."""

    seed: int = 20260601
    rollout: RolloutRuntimeConfig = field(default_factory=RolloutRuntimeConfig)
    include_hidden: bool = True
    batch: PPORolloutBatchConfig = field(default_factory=PPORolloutBatchConfig)
    training: PPOModelTrainingConfig = field(default_factory=_default_training_config)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def resolved_batch_config(self) -> PPORolloutBatchConfig:
        """Return batch settings with rollout token limits filled in."""

        return replace(
            self.batch,
            max_prompt_tokens=(
                self.batch.max_prompt_tokens
                if self.batch.max_prompt_tokens is not None
                else self.rollout.max_prompt_tokens
            ),
            max_response_tokens=(
                self.batch.max_response_tokens
                if self.batch.max_response_tokens is not None
                else self.rollout.max_response_tokens
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "rollout": self.rollout.to_dict(),
            "include_hidden": self.include_hidden,
            "batch": self.resolved_batch_config().to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class PPORolloutTrainingCycleResult:
    """Rollout collection, PPO batch, and training result for one cycle."""

    config: PPORolloutTrainingCycleConfig
    rollouts: tuple[RolloutRecord, ...]
    rollout_batch: PPORolloutBatchResult
    training: PPOModelTrainingResult
    rollouts_path: str | None = None
    metrics_path: str | None = None
    checkpoint_path: str | None = None
    state_dir: str | None = None

    @property
    def rollout_count(self) -> int:
        return len(self.rollouts)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "rollout_count": self.rollout_count,
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
            "rollout_batch": self.rollout_batch.to_dict(),
            "training": self.training.to_dict(),
            "artifacts": {
                "rollouts_path": self.rollouts_path,
                "metrics_path": self.metrics_path,
                "checkpoint_path": self.checkpoint_path,
                "state_dir": self.state_dir,
            },
        }


def run_ppo_rollout_training_cycle(
    tasks: Sequence[TaskSpec],
    *,
    generator: CodeGenerator,
    prompt_template: PromptTemplate,
    tokenizer: TokenizerEngine,
    policy_model: Any,
    value_model: Any | None = None,
    old_policy_model: Any | None = None,
    old_value_model: Any | None = None,
    reference_model: Any | None = None,
    value_estimates: PPOValueEstimateProvider = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scorer: RewardScorer | None = None,
    runner: SandboxedTestRunner | None = None,
    config: PPORolloutTrainingCycleConfig | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    rollouts_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    state_dir: str | Path | None = None,
) -> PPORolloutTrainingCycleResult:
    """Collect rollouts and run one PPO model-training update."""

    task_list = list(tasks)
    if not task_list:
        raise ValueError("PPO rollout training cycle requires at least one task")

    cycle_config = config or PPORolloutTrainingCycleConfig()
    rollouts = generate_rollouts(
        task_list,
        generator=generator,
        prompt_template=prompt_template,
        samples_per_task=cycle_config.rollout.samples_per_task,
        seed=cycle_config.seed,
        max_new_tokens=cycle_config.rollout.max_response_tokens,
        temperature=cycle_config.rollout.temperature,
        top_p=cycle_config.rollout.top_p,
        include_hidden=cycle_config.include_hidden,
        scorer=scorer,
        runner=runner,
        metadata={
            "training_cycle": "ppo_rollout_training",
            "samples_per_task": cycle_config.rollout.samples_per_task,
            **(metadata or {}),
        },
    )
    if rollouts_path is not None:
        write_rollouts_jsonl(rollouts, rollouts_path)

    rollout_batch = build_ppo_training_batch_from_rollouts(
        rollouts,
        tokenizer,
        config=cycle_config.resolved_batch_config(),
    )
    training = run_ppo_model_training(
        (rollout_batch.batch,),
        policy_model=policy_model,
        value_model=value_model,
        old_policy_model=old_policy_model,
        old_value_model=old_value_model,
        reference_model=reference_model,
        value_estimates=value_estimates,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cycle_config.training,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        state_dir=state_dir,
    )
    return PPORolloutTrainingCycleResult(
        config=cycle_config,
        rollouts=tuple(rollouts),
        rollout_batch=rollout_batch,
        training=training,
        rollouts_path=str(rollouts_path) if rollouts_path is not None else None,
        metrics_path=str(metrics_path) if metrics_path is not None else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        state_dir=str(state_dir) if state_dir is not None else None,
    )
