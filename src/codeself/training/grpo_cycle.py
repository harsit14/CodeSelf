"""One-cycle GRPO orchestration from rollouts to model updates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from codeself.agent.generation import CodeGenerator
from codeself.agent.loop import write_agent_traces_jsonl
from codeself.agent.prompts import PromptTemplate
from codeself.agent.rollouts import RolloutRecord, generate_rollouts, write_rollouts_jsonl
from codeself.agent.self_debug_rollouts import generate_self_debug_rollouts
from codeself.datasets import TaskSpec
from codeself.execution import SandboxedTestRunner
from codeself.rewards import RewardScorer
from codeself.training.common import RolloutRuntimeConfig, TokenizerEngine
from codeself.training.grpo_loss import GRPOLossConfig
from codeself.training.grpo_rollouts import (
    GRPORolloutBatchConfig,
    GRPORolloutBatchResult,
    build_grpo_training_batch_from_rollouts,
)
from codeself.training.grpo_trainer import (
    GRPOModelTrainingConfig,
    GRPOModelTrainingResult,
    run_grpo_model_training,
)
from codeself.training.self_debug import SelfDebugCollectionConfig


def _default_training_config() -> GRPOModelTrainingConfig:
    return GRPOModelTrainingConfig(loss=GRPOLossConfig(kl_beta=0.0))


@dataclass(frozen=True)
class GRPORolloutTrainingCycleConfig:
    """Config for one collect -> batch -> optimize GRPO cycle."""

    seed: int = 20260601
    rollout: RolloutRuntimeConfig = field(default_factory=RolloutRuntimeConfig)
    rollout_mode: str = "direct"
    self_debug: SelfDebugCollectionConfig = field(default_factory=SelfDebugCollectionConfig)
    include_hidden: bool = True
    batch: GRPORolloutBatchConfig = field(default_factory=GRPORolloutBatchConfig)
    training: GRPOModelTrainingConfig = field(default_factory=_default_training_config)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.rollout_mode not in {"direct", "self_debug"}:
            raise ValueError("rollout_mode must be direct or self_debug")

    def resolved_batch_config(self) -> GRPORolloutBatchConfig:
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
            "rollout_mode": self.rollout_mode,
            "self_debug": self.self_debug.to_dict(),
            "include_hidden": self.include_hidden,
            "batch": self.resolved_batch_config().to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class GRPORolloutTrainingCycleResult:
    """Rollout collection, GRPO batch, and training result for one cycle."""

    config: GRPORolloutTrainingCycleConfig
    rollouts: tuple[RolloutRecord, ...]
    rollout_batch: GRPORolloutBatchResult
    training: GRPOModelTrainingResult
    rollouts_path: str | None = None
    metrics_path: str | None = None
    checkpoint_path: str | None = None
    state_dir: str | None = None
    traces_path: str | None = None

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
                "traces_path": self.traces_path,
            },
        }


def run_grpo_rollout_training_cycle(
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
    config: GRPORolloutTrainingCycleConfig | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    rollouts_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    state_dir: str | Path | None = None,
    traces_path: str | Path | None = None,
) -> GRPORolloutTrainingCycleResult:
    """Collect grouped rollouts and run one GRPO model-training update."""

    task_list = list(tasks)
    if not task_list:
        raise ValueError("GRPO rollout training cycle requires at least one task")

    cycle_config = config or GRPORolloutTrainingCycleConfig()
    rollouts, written_traces_path = _collect_rollouts(
        task_list,
        generator=generator,
        prompt_template=prompt_template,
        scorer=scorer,
        runner=runner,
        config=cycle_config,
        metadata={
            "training_cycle": "grpo_rollout_training",
            "group_size": cycle_config.rollout.samples_per_task,
            **(metadata or {}),
        },
        traces_path=traces_path,
    )
    if rollouts_path is not None:
        write_rollouts_jsonl(rollouts, rollouts_path)

    rollout_batch = build_grpo_training_batch_from_rollouts(
        rollouts,
        tokenizer,
        config=cycle_config.resolved_batch_config(),
    )
    training = run_grpo_model_training(
        (rollout_batch.batch,),
        policy_model=policy_model,
        old_policy_model=old_policy_model,
        reference_model=reference_model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cycle_config.training,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        state_dir=state_dir,
    )
    return GRPORolloutTrainingCycleResult(
        config=cycle_config,
        rollouts=tuple(rollouts),
        rollout_batch=rollout_batch,
        training=training,
        rollouts_path=str(rollouts_path) if rollouts_path is not None else None,
        metrics_path=str(metrics_path) if metrics_path is not None else None,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        state_dir=str(state_dir) if state_dir is not None else None,
        traces_path=written_traces_path,
    )


def _collect_rollouts(
    tasks: list[TaskSpec],
    *,
    generator: CodeGenerator,
    prompt_template: PromptTemplate,
    scorer: RewardScorer | None,
    runner: SandboxedTestRunner | None,
    config: GRPORolloutTrainingCycleConfig,
    metadata: dict[str, str | int | float | bool],
    traces_path: str | Path | None,
) -> tuple[list[RolloutRecord], str | None]:
    if config.rollout_mode == "direct":
        return (
            generate_rollouts(
                tasks,
                generator=generator,
                prompt_template=prompt_template,
                samples_per_task=config.rollout.samples_per_task,
                seed=config.seed,
                max_new_tokens=config.rollout.max_response_tokens,
                temperature=config.rollout.temperature,
                top_p=config.rollout.top_p,
                include_hidden=config.include_hidden,
                scorer=scorer,
                runner=runner,
                metadata={"rollout_mode": "direct", **metadata},
            ),
            None,
        )
    result = generate_self_debug_rollouts(
        tasks,
        generator=generator,
        prompt_template=prompt_template,
        samples_per_task=config.rollout.samples_per_task,
        seed=config.seed,
        max_new_tokens=config.rollout.max_response_tokens,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        include_hidden=config.include_hidden,
        max_revisions=config.self_debug.max_revisions,
        revision_strategy=config.self_debug.revision_strategy,
        revision_prompt_template=config.self_debug.revision_prompt_template,
        revision_reward_discount=config.self_debug.revision_reward_discount,
        use_rule_based_repair=config.self_debug.revision_strategy == "rule_based",
        scorer=scorer,
        runner=runner,
        metadata=metadata,
    )
    if traces_path is not None:
        write_agent_traces_jsonl(list(result.traces), traces_path)
    return list(result.records), str(traces_path) if traces_path is not None else None
