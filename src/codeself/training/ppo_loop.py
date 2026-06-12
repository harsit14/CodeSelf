"""Minimal PPO training loop over prepared tensor batches."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import Any

from codeself.training.common import TrainingBatch
from codeself.training.ppo_step import (
    PPOOptimizerStepConfig,
    PPOOptimizerStepResult,
    run_ppo_optimizer_step,
)
from codeself.training.ppo_torch import PPOTensorBatch

PPOTensorBatchBuilder = Callable[[TrainingBatch], PPOTensorBatch]


@dataclass(frozen=True)
class PPOTrainingLoopConfig:
    """Settings for a minimal PPO tensor training loop."""

    optimizer_step: PPOOptimizerStepConfig = field(default_factory=PPOOptimizerStepConfig)
    max_batches: int | None = None
    step_scheduler: bool = True

    def __post_init__(self) -> None:
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when set")

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer_step": self.optimizer_step.to_dict(),
            "max_batches": self.max_batches,
            "step_scheduler": self.step_scheduler,
        }


@dataclass(frozen=True)
class PPOTrainingLoopStep:
    """Metrics for one microbatch in a PPO training loop."""

    step: int
    accumulation_index: int
    accumulation_size: int
    optimizer_step_index: int | None
    scheduler_step: bool
    metrics: PPOOptimizerStepResult

    @property
    def optimizer_step(self) -> bool:
        return self.metrics.optimizer_step

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "accumulation_index": self.accumulation_index,
            "accumulation_size": self.accumulation_size,
            "optimizer_step_index": self.optimizer_step_index,
            "optimizer_step": self.optimizer_step,
            "scheduler_step": self.scheduler_step,
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class PPOTrainingLoopResult:
    """Summary of a minimal PPO training loop run."""

    config: PPOTrainingLoopConfig
    steps: tuple[PPOTrainingLoopStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("PPOTrainingLoopResult requires at least one step")

    @property
    def microbatch_count(self) -> int:
        return len(self.steps)

    @property
    def optimizer_step_count(self) -> int:
        return sum(1 for step in self.steps if step.optimizer_step)

    @property
    def mean_loss(self) -> float:
        return fmean(step.metrics.loss for step in self.steps)

    @property
    def mean_policy_loss(self) -> float:
        return fmean(step.metrics.policy_loss for step in self.steps)

    @property
    def mean_value_loss(self) -> float:
        return fmean(step.metrics.value_loss for step in self.steps)

    @property
    def mean_entropy_loss(self) -> float:
        return fmean(step.metrics.entropy_loss for step in self.steps)

    @property
    def mean_kl_loss(self) -> float:
        return fmean(step.metrics.kl_loss for step in self.steps)

    @property
    def total_response_tokens(self) -> int:
        return sum(step.metrics.total_response_tokens for step in self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "microbatch_count": self.microbatch_count,
            "optimizer_step_count": self.optimizer_step_count,
            "mean_loss": self.mean_loss,
            "mean_policy_loss": self.mean_policy_loss,
            "mean_value_loss": self.mean_value_loss,
            "mean_entropy_loss": self.mean_entropy_loss,
            "mean_kl_loss": self.mean_kl_loss,
            "total_response_tokens": self.total_response_tokens,
            "steps": [step.to_dict() for step in self.steps],
        }


def run_ppo_training_loop(
    batches: Iterable[TrainingBatch],
    *,
    tensor_batch_builder: PPOTensorBatchBuilder,
    optimizer: Any,
    scheduler: Any | None = None,
    config: PPOTrainingLoopConfig | None = None,
) -> PPOTrainingLoopResult:
    """Run a minimal PPO training loop over prepared training batches."""

    loop_config = config or PPOTrainingLoopConfig()
    selected_batches = _select_batches(batches, loop_config.max_batches)
    if not selected_batches:
        raise ValueError("run_ppo_training_loop requires at least one batch")

    steps: list[PPOTrainingLoopStep] = []
    optimizer_step_index = 0
    global_step = 0
    accumulation_size = loop_config.optimizer_step.gradient_accumulation_steps
    for group in _chunks(selected_batches, accumulation_size):
        effective_accumulation_size = len(group)
        for index, batch in enumerate(group, start=1):
            global_step += 1
            should_step = index == effective_accumulation_size
            step_config = replace(
                loop_config.optimizer_step,
                gradient_accumulation_steps=effective_accumulation_size,
                zero_grad=index == 1 and loop_config.optimizer_step.zero_grad,
                step_optimizer=should_step and loop_config.optimizer_step.step_optimizer,
            )
            tensor_batch = tensor_batch_builder(batch)
            metrics = run_ppo_optimizer_step(
                tensor_batch,
                optimizer=optimizer,
                config=step_config,
            )
            scheduler_step = False
            if scheduler is not None and should_step and loop_config.step_scheduler:
                scheduler.step()
                scheduler_step = True
            if metrics.optimizer_step:
                optimizer_step_index += 1
                current_optimizer_step = optimizer_step_index
            else:
                current_optimizer_step = None
            steps.append(
                PPOTrainingLoopStep(
                    step=global_step,
                    accumulation_index=index,
                    accumulation_size=effective_accumulation_size,
                    optimizer_step_index=current_optimizer_step,
                    scheduler_step=scheduler_step,
                    metrics=metrics,
                )
            )

    return PPOTrainingLoopResult(config=loop_config, steps=tuple(steps))


def run_ppo_microbatched_training_loop(
    batches: Iterable[TrainingBatch],
    *,
    tensor_batch_builder: PPOTensorBatchBuilder,
    microbatch_size: int,
    optimizer: Any,
    scheduler: Any | None = None,
    config: PPOTrainingLoopConfig | None = None,
) -> PPOTrainingLoopResult:
    """Run PPO training with microbatched forward+backward accumulation.

    Like the GRPO microbatched loop: each logical batch is split into
    ``microbatch_size``-sample microbatches forwarded and back-propagated one
    at a time, so the live autograd graph stays bounded by a microbatch. This
    keeps real-vocabulary value-head models within MPS/GPU memory.
    """

    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")
    loop_config = config or PPOTrainingLoopConfig()
    selected_batches = _select_batches(batches, loop_config.max_batches)
    if not selected_batches:
        raise ValueError("run_ppo_microbatched_training_loop requires at least one batch")

    base_step = loop_config.optimizer_step
    steps: list[PPOTrainingLoopStep] = []
    optimizer_step_index = 0
    global_step = 0
    for batch in selected_batches:
        microbatches = _split_training_batch(batch, microbatch_size)
        accumulation_size = len(microbatches)
        for index, microbatch in enumerate(microbatches, start=1):
            global_step += 1
            should_step = index == accumulation_size
            step_config = replace(
                base_step,
                gradient_accumulation_steps=accumulation_size,
                zero_grad=index == 1 and base_step.zero_grad,
                step_optimizer=should_step and base_step.step_optimizer,
            )
            tensor_batch = tensor_batch_builder(microbatch)
            metrics = run_ppo_optimizer_step(
                tensor_batch,
                optimizer=optimizer,
                config=step_config,
            )
            scheduler_step = False
            if scheduler is not None and should_step and loop_config.step_scheduler:
                scheduler.step()
                scheduler_step = True
            if metrics.optimizer_step:
                optimizer_step_index += 1
                current_optimizer_step: int | None = optimizer_step_index
            else:
                current_optimizer_step = None
            steps.append(
                PPOTrainingLoopStep(
                    step=global_step,
                    accumulation_index=index,
                    accumulation_size=accumulation_size,
                    optimizer_step_index=current_optimizer_step,
                    scheduler_step=scheduler_step,
                    metrics=metrics,
                )
            )

    return PPOTrainingLoopResult(config=loop_config, steps=tuple(steps))


def _split_training_batch(
    batch: TrainingBatch,
    microbatch_size: int,
) -> list[TrainingBatch]:
    samples = list(batch.samples)
    if microbatch_size <= 0 or microbatch_size >= len(samples):
        return [batch]
    return [
        TrainingBatch(tuple(samples[start : start + microbatch_size]))
        for start in range(0, len(samples), microbatch_size)
    ]


def _select_batches(
    batches: Iterable[TrainingBatch],
    max_batches: int | None,
) -> tuple[TrainingBatch, ...]:
    selected: list[TrainingBatch] = []
    for batch in batches:
        selected.append(batch)
        if max_batches is not None and len(selected) >= max_batches:
            break
    return tuple(selected)


def _chunks(
    batches: tuple[TrainingBatch, ...],
    chunk_size: int,
) -> Iterable[tuple[TrainingBatch, ...]]:
    for start in range(0, len(batches), chunk_size):
        yield batches[start : start + chunk_size]
