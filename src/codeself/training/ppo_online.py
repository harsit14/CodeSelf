"""Multi-cycle PPO online training orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from codeself.agent.generation import CodeGenerator
from codeself.agent.prompts import PromptTemplate
from codeself.agent.rollouts import RolloutRecord, generate_rollouts, write_rollouts_jsonl
from codeself.datasets import TaskSpec
from codeself.evaluation import EvaluationSummary, evaluate_rollouts, write_evaluation_report
from codeself.execution import SandboxedTestRunner
from codeself.rewards import RewardScorer
from codeself.training.common import TokenizerEngine
from codeself.training.grpo_torch import require_torch
from codeself.training.ppo_cycle import (
    PPORolloutTrainingCycleConfig,
    PPORolloutTrainingCycleResult,
    run_ppo_rollout_training_cycle,
)
from codeself.training.ppo_trainer import PPOValueEstimateProvider


@dataclass(frozen=True)
class PPOOnlineEvaluationConfig:
    """Dev-evaluation settings for online PPO cycles."""

    samples_per_task: int = 1
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    include_hidden: bool = True
    ks: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        if self.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive when set")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not self.ks or any(k <= 0 for k in self.ks):
            raise ValueError("ks must contain positive integers")

    def to_dict(self) -> dict[str, object]:
        return {
            "samples_per_task": self.samples_per_task,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "include_hidden": self.include_hidden,
            "ks": list(self.ks),
        }


@dataclass(frozen=True)
class PPOOnlineTrainingConfig:
    """Config for repeated PPO rollout-training cycles."""

    cycles: int = 1
    cycle: PPORolloutTrainingCycleConfig = field(
        default_factory=PPORolloutTrainingCycleConfig
    )
    seed_stride: int = 1
    sync_old_policy_before_cycle: bool = True
    sync_old_value_before_cycle: bool = True
    evaluation: PPOOnlineEvaluationConfig | None = None

    def __post_init__(self) -> None:
        if self.cycles <= 0:
            raise ValueError("cycles must be positive")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")

    def cycle_config_for(self, cycle: int) -> PPORolloutTrainingCycleConfig:
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
            "sync_old_value_before_cycle": self.sync_old_value_before_cycle,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }


@dataclass(frozen=True)
class PPOOnlineEvaluationResult:
    """Evaluation rollouts and summary for one online PPO cycle."""

    cycle: int
    seed: int
    rollouts: tuple[RolloutRecord, ...]
    summary: EvaluationSummary
    rollouts_path: str | None = None
    evaluation_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "cycle": self.cycle,
            "seed": self.seed,
            "rollout_count": len(self.rollouts),
            "summary": self.summary.to_dict(),
            "rollouts_path": self.rollouts_path,
            "evaluation_path": self.evaluation_path,
        }


@dataclass(frozen=True)
class PPOOnlineTrainingStep:
    """One completed online PPO cycle."""

    cycle: int
    seed: int
    result: PPORolloutTrainingCycleResult
    old_policy_synced: bool = False
    old_value_synced: bool = False
    evaluation: PPOOnlineEvaluationResult | None = None

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
            "old_value_synced": self.old_value_synced,
            "rollout_count": self.rollout_count,
            "records_used": self.records_used,
            "mean_reward": self.mean_reward,
            "optimizer_step_count": self.optimizer_step_count,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "result": self.result.to_dict(),
        }


@dataclass(frozen=True)
class PPOOnlineTrainingResult:
    """Result summary for repeated PPO online training cycles."""

    config: PPOOnlineTrainingConfig
    steps: tuple[PPOOnlineTrainingStep, ...]
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


def run_ppo_online_training(
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
    eval_tasks: Sequence[TaskSpec] | None = None,
    eval_generator: CodeGenerator | None = None,
    eval_prompt_template: PromptTemplate | None = None,
    eval_scorer: RewardScorer | None = None,
    eval_runner: SandboxedTestRunner | None = None,
    scorer: RewardScorer | None = None,
    runner: SandboxedTestRunner | None = None,
    config: PPOOnlineTrainingConfig | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
    artifact_dir: str | Path | None = None,
) -> PPOOnlineTrainingResult:
    """Run repeated collect -> execute -> score -> optimize PPO cycles."""

    task_list = list(tasks)
    if not task_list:
        raise ValueError("PPO online training requires at least one task")

    online_config = config or PPOOnlineTrainingConfig()
    eval_task_list = list(eval_tasks or ())
    eval_config = online_config.evaluation
    if eval_task_list and eval_config is None:
        eval_config = PPOOnlineEvaluationConfig()
    output_dir = Path(artifact_dir) if artifact_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    active_optimizer = optimizer or _build_optimizer(policy_model, value_model, online_config)
    steps: list[PPOOnlineTrainingStep] = []
    for cycle in range(1, online_config.cycles + 1):
        cycle_config = online_config.cycle_config_for(cycle)
        cycle_dir = output_dir / f"cycle_{cycle:04d}" if output_dir is not None else None
        if cycle_dir is not None:
            cycle_dir.mkdir(parents=True, exist_ok=True)
        old_policy_synced = False
        old_value_synced = False
        if old_policy_model is not None and online_config.sync_old_policy_before_cycle:
            _sync_model_state(policy_model, old_policy_model, source_name="policy_model")
            old_policy_synced = True
        if old_value_model is not None and online_config.sync_old_value_before_cycle:
            _sync_model_state(
                value_model or policy_model,
                old_value_model,
                source_name="value_model",
            )
            old_value_synced = True
        result = run_ppo_rollout_training_cycle(
            task_list,
            generator=generator,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            policy_model=policy_model,
            value_model=value_model,
            old_policy_model=old_policy_model,
            old_value_model=old_value_model,
            reference_model=reference_model,
            value_estimates=value_estimates,
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
            traces_path=(
                cycle_dir / "self_debug_traces.jsonl"
                if cycle_dir is not None and cycle_config.rollout_mode == "self_debug"
                else None
            ),
        )
        evaluation = (
            _run_cycle_evaluation(
                cycle,
                cycle_config.seed,
                eval_task_list,
                generator=eval_generator or generator,
                prompt_template=eval_prompt_template or prompt_template,
                scorer=eval_scorer or scorer,
                runner=eval_runner or runner,
                config=eval_config,
                max_new_tokens_fallback=cycle_config.rollout.max_response_tokens,
                cycle_dir=cycle_dir,
                metadata=metadata,
            )
            if eval_task_list and eval_config is not None
            else None
        )
        steps.append(
            PPOOnlineTrainingStep(
                cycle=cycle,
                seed=cycle_config.seed,
                result=result,
                old_policy_synced=old_policy_synced,
                old_value_synced=old_value_synced,
                evaluation=evaluation,
            )
        )

    return PPOOnlineTrainingResult(
        config=online_config,
        steps=tuple(steps),
        artifact_dir=str(output_dir) if output_dir is not None else None,
    )


def _run_cycle_evaluation(
    cycle: int,
    seed: int,
    tasks: list[TaskSpec],
    *,
    generator: CodeGenerator,
    prompt_template: PromptTemplate,
    scorer: RewardScorer | None,
    runner: SandboxedTestRunner | None,
    config: PPOOnlineEvaluationConfig,
    max_new_tokens_fallback: int,
    cycle_dir: Path | None,
    metadata: dict[str, str | int | float | bool] | None,
) -> PPOOnlineEvaluationResult:
    rollouts = generate_rollouts(
        tasks,
        generator=generator,
        prompt_template=prompt_template,
        samples_per_task=config.samples_per_task,
        seed=seed,
        max_new_tokens=config.max_new_tokens or max_new_tokens_fallback,
        temperature=config.temperature,
        top_p=config.top_p,
        include_hidden=config.include_hidden,
        scorer=scorer,
        runner=runner,
        metadata={
            "online_cycle": cycle,
            "online_evaluation": True,
            **(metadata or {}),
        },
    )
    summary = evaluate_rollouts(rollouts, ks=config.ks)
    rollouts_path = cycle_dir / "eval_rollouts.jsonl" if cycle_dir is not None else None
    evaluation_path = cycle_dir / "evaluation.json" if cycle_dir is not None else None
    if rollouts_path is not None:
        write_rollouts_jsonl(rollouts, rollouts_path)
    if evaluation_path is not None:
        write_evaluation_report(summary, evaluation_path)
    return PPOOnlineEvaluationResult(
        cycle=cycle,
        seed=seed,
        rollouts=tuple(rollouts),
        summary=summary,
        rollouts_path=str(rollouts_path) if rollouts_path is not None else None,
        evaluation_path=str(evaluation_path) if evaluation_path is not None else None,
    )


def _build_optimizer(
    policy_model: Any,
    value_model: Any | None,
    config: PPOOnlineTrainingConfig,
) -> Any:
    torch = require_torch()
    parameters = _trainable_parameters_for_models(policy_model, value_model)
    if not parameters:
        raise ValueError("policy_model/value_model must expose trainable parameters")
    optimizer_config = config.cycle.training.optimizer
    return torch.optim.AdamW(
        parameters,
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
    )


def _sync_model_state(source_model: Any, target_model: Any, *, source_name: str) -> None:
    if not hasattr(source_model, "state_dict"):
        raise TypeError(f"{source_name} must define state_dict() to sync old model state")
    if not hasattr(target_model, "load_state_dict"):
        raise TypeError("old policy/value model must define load_state_dict()")
    target_model.load_state_dict(source_model.state_dict())


def _trainable_parameters_for_models(*models: Any | None) -> list[Any]:
    parameters: list[Any] = []
    seen: set[int] = set()
    for model in models:
        if model is None:
            continue
        if not hasattr(model, "parameters"):
            raise TypeError("policy_model/value_model must define parameters()")
        for parameter in model.parameters():
            if not bool(getattr(parameter, "requires_grad", True)):
                continue
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            parameters.append(parameter)
    return parameters
