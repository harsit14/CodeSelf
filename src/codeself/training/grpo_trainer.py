"""Minimal model-aware GRPO trainer over Torch causal-LM modules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeself.training.callbacks import JsonlMetricWriter, write_checkpoint_manifest
from codeself.training.common import OptimizerConfig, TrainingBatch
from codeself.training.grpo_loop import GRPOTrainingLoopConfig, GRPOTrainingLoopResult
from codeself.training.grpo_loss import GRPOLossConfig
from codeself.training.grpo_model import build_grpo_tensor_batch_from_model
from codeself.training.grpo_step import GRPOOptimizerStepConfig
from codeself.training.grpo_torch import require_torch


@dataclass(frozen=True)
class GRPOModelTrainingConfig:
    """Config for the minimal model-aware GRPO trainer."""

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: GRPOLossConfig = field(default_factory=GRPOLossConfig)
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

    def loop_config(self) -> GRPOTrainingLoopConfig:
        return GRPOTrainingLoopConfig(
            optimizer_step=GRPOOptimizerStepConfig(
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
class GRPOCheckpointArtifact:
    """One state artifact written by a model-training checkpoint."""

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
class GRPOModelTrainingResult:
    """Result and checkpoint metadata for a minimal GRPO model run."""

    config: GRPOModelTrainingConfig
    loop: GRPOTrainingLoopResult
    policy_model_class: str
    optimizer_class: str
    old_policy_model_class: str | None = None
    reference_model_class: str | None = None
    checkpoint_artifacts: tuple[GRPOCheckpointArtifact, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "loop": self.loop.to_dict(),
            "policy_model_class": self.policy_model_class,
            "optimizer_class": self.optimizer_class,
            "old_policy_model_class": self.old_policy_model_class,
            "reference_model_class": self.reference_model_class,
            "checkpoint_artifacts": [
                artifact.to_dict() for artifact in self.checkpoint_artifacts
            ],
        }

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "kind": "grpo_model_training_checkpoint",
            "has_real_model_weights": True,
            "has_state_artifacts": bool(self.checkpoint_artifacts),
            "config": self.config.to_dict(),
            "model": {
                "policy_class": self.policy_model_class,
                "old_policy_class": self.old_policy_model_class,
                "reference_class": self.reference_model_class,
            },
            "optimizer_class": self.optimizer_class,
            "artifacts": [artifact.to_dict() for artifact in self.checkpoint_artifacts],
            "loop": self.loop.to_dict(),
            "notes": (
                "Minimal GRPO model-training manifest. When state artifacts are "
                "present, their SHA-256 checksums are recorded here."
            ),
        }


def run_grpo_model_training(
    batches: tuple[TrainingBatch, ...],
    *,
    policy_model: Any,
    old_policy_model: Any | None = None,
    reference_model: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    config: GRPOModelTrainingConfig | None = None,
    metrics_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    state_dir: str | Path | None = None,
) -> GRPOModelTrainingResult:
    """Run the minimal model-aware GRPO training loop."""

    torch = require_torch()
    train_config = config or GRPOModelTrainingConfig()
    _configure_model_modes(policy_model, old_policy_model, reference_model)
    _move_model(policy_model, train_config.device)
    _move_model(old_policy_model, train_config.device)
    _move_model(reference_model, train_config.device)
    active_optimizer = optimizer or _build_optimizer(torch, policy_model, train_config.optimizer)
    tensor_dtype = _torch_dtype(torch, train_config.dtype)

    def tensor_batch_builder(batch: TrainingBatch) -> Any:
        return build_grpo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            old_policy_model=old_policy_model,
            reference_model=reference_model,
            device=train_config.device,
            dtype=tensor_dtype,
            pad_token_id=train_config.pad_token_id,
        )

    from codeself.training.grpo_loop import run_grpo_training_loop

    loop_result = run_grpo_training_loop(
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
            optimizer=active_optimizer,
            scheduler=scheduler,
        )
        if state_dir is not None
        else ()
    )
    result = GRPOModelTrainingResult(
        config=train_config,
        loop=loop_result,
        policy_model_class=policy_model.__class__.__name__,
        old_policy_model_class=old_policy_model.__class__.__name__ if old_policy_model else None,
        reference_model_class=reference_model.__class__.__name__ if reference_model else None,
        optimizer_class=active_optimizer.__class__.__name__,
        checkpoint_artifacts=checkpoint_artifacts,
    )
    if metrics_path is not None:
        _write_metrics(metrics_path, result)
    if checkpoint_path is not None:
        write_checkpoint_manifest(checkpoint_path, result.checkpoint_payload())
    return result


def _build_optimizer(torch: Any, policy_model: Any, config: OptimizerConfig) -> Any:
    parameters = _trainable_parameters(policy_model)
    if not parameters:
        raise ValueError("policy_model must expose at least one trainable parameter")
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
    optimizer: Any,
    scheduler: Any | None,
) -> tuple[GRPOCheckpointArtifact, ...]:
    output_dir = Path(state_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[GRPOCheckpointArtifact] = []

    if not hasattr(policy_model, "state_dict"):
        raise TypeError("policy_model must define state_dict() to write state checkpoints")
    policy_path = output_dir / "policy_model.pt"
    torch.save(policy_model.state_dict(), policy_path)
    artifacts.append(_checkpoint_artifact("policy_model_state", policy_path))

    if hasattr(optimizer, "state_dict"):
        optimizer_path = output_dir / "optimizer.pt"
        torch.save(optimizer.state_dict(), optimizer_path)
        artifacts.append(_checkpoint_artifact("optimizer_state", optimizer_path))

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler_path = output_dir / "scheduler.pt"
        torch.save(scheduler.state_dict(), scheduler_path)
        artifacts.append(_checkpoint_artifact("scheduler_state", scheduler_path))

    return tuple(artifacts)


def _checkpoint_artifact(kind: str, path: Path) -> GRPOCheckpointArtifact:
    return GRPOCheckpointArtifact(
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


def _trainable_parameters(model: Any) -> list[Any]:
    if not hasattr(model, "parameters"):
        raise TypeError("policy_model must define parameters()")
    return [
        parameter
        for parameter in model.parameters()
        if bool(getattr(parameter, "requires_grad", True))
    ]


def _configure_model_modes(
    policy_model: Any,
    old_policy_model: Any | None,
    reference_model: Any | None,
) -> None:
    if hasattr(policy_model, "train"):
        policy_model.train()
    for model in (old_policy_model, reference_model):
        if model is not None and hasattr(model, "eval"):
            model.eval()


def _move_model(model: Any | None, device: str | None) -> None:
    if model is not None and device is not None and hasattr(model, "to"):
        model.to(device)


def _write_metrics(path: str | Path, result: GRPOModelTrainingResult) -> None:
    writer = JsonlMetricWriter(path)
    for step in result.loop.steps:
        writer.write(
            {
                "phase": "grpo_model_training",
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
