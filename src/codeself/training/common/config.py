"""Shared training-core config records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from codeself.config import config_get
from codeself.training.common.backends import get_backend_spec


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Model/tokenizer runtime settings for trainable backends."""

    name: str
    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "bf16"
    device: str = "auto"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = True
    use_lora: bool = True
    lora_rank: int = 16
    full_finetune: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("model name must not be empty")
        if self.dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("dtype must be fp32, fp16, or bf16")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.full_finetune and self.use_lora:
            raise ValueError("full_finetune and use_lora cannot both be true")

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer or self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tokenizer": self.tokenizer_name,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "dtype": self.dtype,
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "full_finetune": self.full_finetune,
        }


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer-level settings shared by trainable backends."""

    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass(frozen=True)
class RolloutRuntimeConfig:
    """Sampling and sequence-shape settings for model rollouts."""

    group_size: int = 4
    samples_per_task: int = 4
    max_prompt_tokens: int = 1024
    max_response_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.samples_per_task <= 0:
            raise ValueError("samples_per_task must be positive")
        if self.max_prompt_tokens <= 0 or self.max_response_tokens <= 0:
            raise ValueError("token limits must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "samples_per_task": self.samples_per_task,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_response_tokens": self.max_response_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass(frozen=True)
class TrainingCoreConfig:
    """Top-level config for future trainable GRPO/PPO backends."""

    algorithm: str
    backend: str = "from_scratch"
    seed: int = 20260601
    max_steps: int = 100
    kl_beta: float = 0.05
    clip_epsilon: float = 0.2
    model: ModelRuntimeConfig = field(default_factory=lambda: ModelRuntimeConfig(name="unset"))
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    rollout: RolloutRuntimeConfig = field(default_factory=RolloutRuntimeConfig)

    def __post_init__(self) -> None:
        if self.algorithm not in {"grpo", "ppo"}:
            raise ValueError("algorithm must be grpo or ppo")
        get_backend_spec(self.backend)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        if self.clip_epsilon <= 0:
            raise ValueError("clip_epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "backend": self.backend,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "kl_beta": self.kl_beta,
            "clip_epsilon": self.clip_epsilon,
            "model": self.model.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "rollout": self.rollout.to_dict(),
        }


def build_training_core_config(config: dict[str, Any]) -> TrainingCoreConfig:
    """Build a `TrainingCoreConfig` from a nested experiment config mapping."""

    model_name = (
        config_get(config, "model.name")
        or config_get(config, "model.model_name_or_path")
        or config_get(config, "model.config")
        or "unset"
    )
    model = ModelRuntimeConfig(
        name=str(model_name),
        tokenizer=_optional_str(config_get(config, "model.tokenizer")),
        revision=_optional_str(config_get(config, "model.revision")),
        tokenizer_revision=_optional_str(config_get(config, "model.tokenizer_revision")),
        dtype=str(config_get(config, "model.dtype", "bf16")),
        device=str(config_get(config, "model.device", "auto")),
        trust_remote_code=bool(config_get(config, "model.trust_remote_code", False)),
        gradient_checkpointing=bool(config_get(config, "model.gradient_checkpointing", True)),
        use_lora=bool(config_get(config, "model.use_lora", True)),
        lora_rank=int(config_get(config, "model.lora_rank", config_get(config, "model.lora_rank", 16))),
        full_finetune=bool(config_get(config, "model.full_finetune", False)),
    )
    optimizer = OptimizerConfig(
        learning_rate=float(config_get(config, "training.learning_rate", 1e-6)),
        weight_decay=float(config_get(config, "training.weight_decay", 0.0)),
        warmup_steps=int(config_get(config, "training.warmup_steps", 0)),
        gradient_accumulation_steps=int(
            config_get(config, "training.gradient_accumulation_steps", 1)
        ),
        max_grad_norm=float(config_get(config, "training.max_grad_norm", 1.0)),
    )
    group_size = int(config_get(config, "rollout.group_size", config_get(config, "generation.group_size", 4)))
    samples_per_task = int(
        config_get(config, "rollout.samples_per_task", config_get(config, "generation.samples_per_task", group_size))
    )
    rollout = RolloutRuntimeConfig(
        group_size=group_size,
        samples_per_task=samples_per_task,
        max_prompt_tokens=int(config_get(config, "rollout.max_prompt_tokens", 1024)),
        max_response_tokens=int(
            config_get(config, "rollout.max_response_tokens", config_get(config, "generation.max_new_tokens", 512))
        ),
        temperature=float(
            config_get(config, "rollout.temperature", config_get(config, "generation.temperature", 0.8))
        ),
        top_p=float(config_get(config, "rollout.top_p", config_get(config, "generation.top_p", 0.95))),
    )
    return TrainingCoreConfig(
        algorithm=str(config_get(config, "training.algorithm", "grpo")).replace("_smoke", ""),
        backend=str(config_get(config, "training.backend", "from_scratch")),
        seed=int(config_get(config, "experiment.seed", 20260601)),
        max_steps=int(config_get(config, "training.max_steps", 100)),
        kl_beta=float(config_get(config, "training.kl_beta", 0.05)),
        clip_epsilon=float(config_get(config, "training.clip_epsilon", 0.2)),
        model=model,
        optimizer=optimizer,
        rollout=rollout,
    )


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
