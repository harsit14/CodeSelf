"""Minimal model-aware PPO trainer over Torch causal-LM modules."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeself.training.callbacks import JsonlMetricWriter, write_checkpoint_manifest
from codeself.training.common import OptimizerConfig, TrainingBatch
from codeself.training.ppo_loop import PPOTrainingLoopConfig, PPOTrainingLoopResult
from codeself.training.ppo_loss import PPOLossConfig, ValueEstimateInput
from codeself.training.ppo_model import build_ppo_tensor_batch_from_model
from codeself.training.ppo_step import PPOOptimizerStepConfig
from codeself.training.grpo_torch import require_torch

PPOValueEstimateProvider = (
    ValueEstimateInput | Callable[[TrainingBatch], ValueEstimateInput | None] | None
)


@dataclass(frozen=True)
class PPOModelTrainingConfig:
    """Config for the minimal model-aware PPO trainer."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: PPOLossConfig = field(default_factory=PPOLossConfig)
    max_batches: int | None = None
    step_scheduler: bool = True
    pad_token_id: int = 0
    device: str | None = None
    dtype: str = "fp32"

    def __post_init__(self) -> None:
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when set")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be non-negative")
        if self.dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("dtype must be fp32, fp16, or bf16")

    def loop_config(self) -> PPOTrainingLoopConfig:
        return PPOTrainingLoopConfig(
            optimizer_step=PPOOptimizerStepConfig(
                loss=self.loss,
                gradient_accumulation_steps=self.optimizer.gradient_accumulation_steps,
                max_grad_norm=self.optimizer.max_grad_norm,
            ),
            max_batches=self.max_batches,
            step_scheduler=self.step_scheduler,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.optimizer.to_dict(),
            "loss": self.loss.to_dict(),
            "max_batches": self.max_batches,
            "step_scheduler": self.step_scheduler,
            "pad_token_id": self.pad_token_id,
            "device": self.device,
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class PPOCheckpointArtifact:
    """One state artifact written by a PPO model-training checkpoint."""

    kind: str
    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class PPOModelTrainingResult:
    """Result and checkpoint metadata for a minimal PPO model run."""

    config: PPOModelTrainingConfig
    loop: PPOTrainingLoopResult
    policy_model_class: str
    optimizer_class: str
    value_model_class: str | None = None
    old_policy_model_class: str | None = None
    old_value_model_class: str | None = None
    reference_model_class: str | None = None
    checkpoint_artifacts: tuple[PPOCheckpointArtifact, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "loop": self.loop.to_dict(),
            "policy_model_class": self.policy_model_class,
            "value_model_class": self.value_model_class,
            "old_policy_model_class": self.old_policy_model_class,
            "old_value_model_class": self.old_value_model_class,
            "reference_model_class": self.reference_model_class,
            "optimizer_class": self.optimizer_class,
            "checkpoint_artifacts": [
                artifact.to_dict() for artifact in self.checkpoint_artifacts
            ],
        }

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "kind": "ppo_model_training_checkpoint",
            "has_real_model_weights": True,
            "has_state_artifacts": bool(self.checkpoint_artifacts),
            "config": self.config.to_dict(),
            "model": {
                "policy_class": self.policy_model_class,
                "value_class": self.value_model_class,
                "old_policy_class": self.old_policy_model_class,
                "old_value_class": self.old_value_model_class,
                "reference_class": self.reference_model_class,
            },
            "optimizer_class": self.optimizer_class,
            "artifacts": [artifact.to_dict() for artifact in self.checkpoint_artifacts],
            "loop": self.loop.to_dict(),
            "notes": (
                "Minimal PPO model-training manifest. When state artifacts are "
                "present, their SHA-256 checksums are recorded here."
            ),
        }


def run_ppo_model_training(
    batches: tuple[TrainingBatch, ...],
    *,
    policy_model: Any,
    value_model: Any | None = None,
    old_policy_model: Any | None = None,
    old_value_model: Any | None = None,
    reference_model: Any | None = None,
    value_estimates: PPOValueEstimateProvider = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    config: PPOModelTrainingConfig | None = None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    state_dir: str | Path | None = None,
) -> PPOModelTrainingResult:
    """Run the minimal model-aware PPO training loop."""

    torch = require_torch()
    train_config = config or PPOModelTrainingConfig()
    _configure_model_modes(
        policy_model,
        value_model,
        old_policy_model,
        old_value_model,
        reference_model,
    )
    _move_model(policy_model, train_config.device)
    _move_model(value_model, train_config.device)
    _move_model(old_policy_model, train_config.device)
    _move_model(old_value_model, train_config.device)
    _move_model(reference_model, train_config.device)
    active_optimizer = optimizer or _build_optimizer(
        torch,
        train_config.optimizer,
        policy_model,
        value_model,
    )
    tensor_dtype = _torch_dtype(torch, train_config.dtype)

    def tensor_batch_builder(batch: TrainingBatch) -> Any:
        return build_ppo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            value_model=value_model,
            old_policy_model=old_policy_model,
            old_value_model=old_value_model,
            reference_model=reference_model,
            value_estimates=_value_estimates_for_batch(value_estimates, batch),
            config=train_config.loss,
            device=train_config.device,
            dtype=tensor_dtype,
            pad_token_id=train_config.pad_token_id,
        )

    from codeself.training.ppo_loop import run_ppo_training_loop

    loop_result = run_ppo_training_loop(
        batches,
        tensor_batch_builder=tensor_batch_builder,
        optimizer=active_optimizer,
        scheduler=scheduler,
        config=train_config.loop_config(),
    )
    checkpoint_artifacts = (
        _write_state_checkpoint(
            torch,
            state_dir,
            policy_model=policy_model,
            value_model=value_model,
            optimizer=active_optimizer,
            scheduler=scheduler,
        )
        if state_dir is not None
        else ()
    )
    result = PPOModelTrainingResult(
        config=train_config,
        loop=loop_result,
        policy_model_class=policy_model.__class__.__name__,
        value_model_class=value_model.__class__.__name__ if value_model else None,
        old_policy_model_class=old_policy_model.__class__.__name__ if old_policy_model else None,
        old_value_model_class=old_value_model.__class__.__name__ if old_value_model else None,
        reference_model_class=reference_model.__class__.__name__ if reference_model else None,
        optimizer_class=active_optimizer.__class__.__name__,
        checkpoint_artifacts=checkpoint_artifacts,
    )
    if metrics_path is not None:
        _write_metrics(metrics_path, result)
    if checkpoint_path is not None:
        write_checkpoint_manifest(checkpoint_path, result.checkpoint_payload())
    return result


def _value_estimates_for_batch(
    value_estimates: PPOValueEstimateProvider,
    batch: TrainingBatch,
) -> ValueEstimateInput | None:
    if callable(value_estimates):
        return value_estimates(batch)
    return value_estimates


def _build_optimizer(torch: Any, config: OptimizerConfig, *models: Any | None) -> Any:
    parameters = _trainable_parameters_for_models(*models)
    if not parameters:
        raise ValueError("policy_model/value_model must expose trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _write_state_checkpoint(
    torch: Any,
    state_dir: str | Path,
    *,
    policy_model: Any,
    value_model: Any | None,
    optimizer: Any,
    scheduler: Any | None,
) -> tuple[PPOCheckpointArtifact, ...]:
    output_dir = Path(state_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[PPOCheckpointArtifact] = []

    artifacts.extend(_write_model_state(torch, output_dir, "policy_model", policy_model))
    if value_model is not None and value_model is not policy_model:
        artifacts.extend(_write_model_state(torch, output_dir, "value_model", value_model))

    if hasattr(optimizer, "state_dict"):
        optimizer_path = output_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        artifacts.append(_checkpoint_artifact("optimizer_state", optimizer_path))

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler_path = output_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path)
        artifacts.append(_checkpoint_artifact("scheduler_state", scheduler_path))

    return tuple(artifacts)


def _write_model_state(
    torch: Any,
    output_dir: Path,
    label: str,
    model: Any,
) -> tuple[PPOCheckpointArtifact, ...]:
    if not hasattr(model, "state_dict"):
        raise TypeError(f"{label} must define state_dict() to write state checkpoints")
    artifacts: list[PPOCheckpointArtifact] = []
    state_path = output_dir / f"{label}.pt"
    torch.save(model.state_dict(), state_path)
    artifacts.append(_checkpoint_artifact(f"{label}_state", state_path))

    if hasattr(model, "save_pretrained"):
        pretrained_dir = output_dir / f"{label}_pretrained"
        model.save_pretrained(pretrained_dir)
        artifacts.extend(
            _checkpoint_artifacts_for_directory(f"{label}_pretrained", pretrained_dir)
        )
    return tuple(artifacts)


def _checkpoint_artifacts_for_directory(
    kind_prefix: str,
    directory: Path,
) -> tuple[PPOCheckpointArtifact, ...]:
    return tuple(
        _checkpoint_artifact(f"{kind_prefix}:{path.relative_to(directory)}", path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    )


def _checkpoint_artifact(kind: str, path: Path) -> PPOCheckpointArtifact:
    return PPOCheckpointArtifact(
        kind=kind,
        path=str(path),
        bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _configure_model_modes(
    policy_model: Any,
    value_model: Any | None,
    old_policy_model: Any | None,
    old_value_model: Any | None,
    reference_model: Any | None,
) -> None:
    if hasattr(policy_model, "train"):
        policy_model.train()
    if (
        value_model is not None
        and value_model is not policy_model
        and hasattr(value_model, "train")
    ):
        value_model.train()
    for model in (old_policy_model, old_value_model, reference_model):
        if model is not None and hasattr(model, "eval"):
            model.eval()


def _move_model(model: Any | None, device: str | None) -> None:
    if model is not None and device is not None and hasattr(model, "to"):
        model.to(device)


def _write_metrics(path: str | Path, result: PPOModelTrainingResult) -> None:
    writer = JsonlMetricWriter(path)
    for step in result.loop.steps:
        writer.write(
            {
                "phase": "ppo_model_training",
                "step": step.step,
                "metrics": step.to_dict(),
            }
        )


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be fp32, fp16, or bf16")
