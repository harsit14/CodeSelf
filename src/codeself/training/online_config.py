"""Config builders for online GRPO/PPO training experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from codeself.config import config_get
from codeself.training.common import OptimizerConfig, RolloutRuntimeConfig
from codeself.training.grpo_cycle import GRPORolloutTrainingCycleConfig
from codeself.training.grpo_loss import GRPOLossConfig
from codeself.training.grpo_online import (
    GRPOOnlineEvaluationConfig,
    GRPOOnlineTrainingConfig,
)
from codeself.training.grpo_rollouts import GRPORolloutBatchConfig
from codeself.training.grpo_trainer import GRPOModelTrainingConfig
from codeself.training.ppo_cycle import PPORolloutTrainingCycleConfig
from codeself.training.ppo_loss import PPOLossConfig
from codeself.training.ppo_online import PPOOnlineEvaluationConfig, PPOOnlineTrainingConfig
from codeself.training.ppo_rollouts import PPORolloutBatchConfig
from codeself.training.ppo_trainer import PPOModelTrainingConfig
from codeself.training.self_debug import SelfDebugCollectionConfig

OnlineAlgorithm = Literal["grpo", "ppo"]
OnlineTrainingConfig = GRPOOnlineTrainingConfig | PPOOnlineTrainingConfig

_MISSING = object()


def resolve_online_algorithm(
    config: dict[str, Any],
    algorithm: str | None = None,
) -> OnlineAlgorithm:
    """Resolve an online-training algorithm from an experiment config."""

    raw = algorithm
    if raw is None:
        raw = str(
            _first_value(
                config,
                "online.algorithm",
                "training.algorithm",
                "algorithm",
                default="grpo",
            )
        )
    normalized = raw.lower().replace("_online", "").replace("_smoke", "")
    if normalized == "grpo":
        return "grpo"
    if normalized == "ppo":
        return "ppo"
    raise ValueError("online training algorithm must be grpo or ppo")


def build_online_training_config(
    config: dict[str, Any],
    *,
    algorithm: str | None = None,
) -> OnlineTrainingConfig:
    """Build an online GRPO/PPO config from a nested experiment mapping."""

    resolved = resolve_online_algorithm(config, algorithm)
    if resolved == "grpo":
        return build_grpo_online_training_config(config)
    return build_ppo_online_training_config(config)


def build_grpo_online_training_config(config: dict[str, Any]) -> GRPOOnlineTrainingConfig:
    """Build a `GRPOOnlineTrainingConfig` from a nested experiment mapping."""

    return GRPOOnlineTrainingConfig(
        cycles=_int_value(config, "online.cycles", "training.cycles", default=1),
        cycle=GRPORolloutTrainingCycleConfig(
            seed=_int_value(config, "cycle.seed", "experiment.seed", default=20260601),
            rollout=_build_rollout_runtime_config(config),
            rollout_mode=_str_value(
                config,
                "cycle.rollout_mode",
                "rollout.mode",
                default="direct",
            ),
            self_debug=_build_self_debug_config(config),
            include_hidden=_bool_value(
                config,
                "cycle.include_hidden",
                "execution.include_hidden",
                default=True,
            ),
            batch=_build_grpo_batch_config(config),
            training=_build_grpo_model_training_config(config),
        ),
        seed_stride=_int_value(
            config,
            "online.seed_stride",
            "training.seed_stride",
            default=1,
        ),
        sync_old_policy_before_cycle=_bool_value(
            config,
            "online.sync_old_policy_before_cycle",
            "training.sync_old_policy_before_cycle",
            default=True,
        ),
        evaluation=_build_grpo_evaluation_config(config),
    )


def build_ppo_online_training_config(config: dict[str, Any]) -> PPOOnlineTrainingConfig:
    """Build a `PPOOnlineTrainingConfig` from a nested experiment mapping."""

    return PPOOnlineTrainingConfig(
        cycles=_int_value(config, "online.cycles", "training.cycles", default=1),
        cycle=PPORolloutTrainingCycleConfig(
            seed=_int_value(config, "cycle.seed", "experiment.seed", default=20260601),
            rollout=_build_rollout_runtime_config(config),
            rollout_mode=_str_value(
                config,
                "cycle.rollout_mode",
                "rollout.mode",
                default="direct",
            ),
            self_debug=_build_self_debug_config(config),
            include_hidden=_bool_value(
                config,
                "cycle.include_hidden",
                "execution.include_hidden",
                default=True,
            ),
            batch=_build_ppo_batch_config(config),
            training=_build_ppo_model_training_config(config),
        ),
        seed_stride=_int_value(
            config,
            "online.seed_stride",
            "training.seed_stride",
            default=1,
        ),
        sync_old_policy_before_cycle=_bool_value(
            config,
            "online.sync_old_policy_before_cycle",
            "training.sync_old_policy_before_cycle",
            default=True,
        ),
        sync_old_value_before_cycle=_bool_value(
            config,
            "online.sync_old_value_before_cycle",
            "training.sync_old_value_before_cycle",
            default=True,
        ),
        evaluation=_build_ppo_evaluation_config(config),
    )


def _build_rollout_runtime_config(config: dict[str, Any]) -> RolloutRuntimeConfig:
    group_size = _int_value(
        config,
        "rollout.group_size",
        "generation.group_size",
        default=4,
    )
    return RolloutRuntimeConfig(
        group_size=group_size,
        samples_per_task=_int_value(
            config,
            "rollout.samples_per_task",
            "generation.samples_per_task",
            default=group_size,
        ),
        max_prompt_tokens=_int_value(config, "rollout.max_prompt_tokens", default=1024),
        max_response_tokens=_int_value(
            config,
            "rollout.max_response_tokens",
            "generation.max_response_tokens",
            "generation.max_new_tokens",
            default=512,
        ),
        temperature=_float_value(
            config,
            "rollout.temperature",
            "generation.temperature",
            default=0.8,
        ),
        top_p=_float_value(config, "rollout.top_p", "generation.top_p", default=0.95),
    )


def _build_self_debug_config(config: dict[str, Any]) -> SelfDebugCollectionConfig:
    strategy = _first_value(config, "self_debug.revision_strategy", default=None)
    if strategy is None:
        use_rule_based_repair = _bool_value(
            config,
            "self_debug.use_rule_based_repair",
            "agent.use_rule_based_repair",
            default=True,
        )
        strategy = "rule_based" if use_rule_based_repair else "none"
    return SelfDebugCollectionConfig(
        max_revisions=_int_value(
            config,
            "self_debug.max_revisions",
            "agent.max_revisions",
            default=1,
        ),
        revision_strategy=str(strategy),
        revision_prompt_template=_str_value(
            config,
            "self_debug.revision_prompt_template",
            "agent.revision_prompt_template",
            default="self_debug_revision_v1",
        ),
        revision_reward_discount=_float_value(
            config,
            "self_debug.revision_reward_discount",
            "agent.revision_reward_discount",
            default=1.0,
        ),
        revision_reward_discount_mode=str(
            _first_value(
                config,
                "self_debug.revision_reward_discount_mode",
                "agent.revision_reward_discount_mode",
                default="positive_only",
            )
        ),
        revision_reward_step_penalty=_float_value(
            config,
            "self_debug.revision_reward_step_penalty",
            "agent.revision_reward_step_penalty",
            default=0.0,
        ),
        revision_max_new_tokens=_optional_int_value(
            _first_value(
                config,
                "self_debug.revision_max_new_tokens",
                "agent.revision_max_new_tokens",
                default=None,
            )
        ),
        revision_temperature=_optional_float_value(
            _first_value(
                config,
                "self_debug.revision_temperature",
                "agent.revision_temperature",
                default=None,
            )
        ),
        revision_top_p=_optional_float_value(
            _first_value(
                config,
                "self_debug.revision_top_p",
                "agent.revision_top_p",
                default=None,
            )
        ),
    )


def _build_optimizer_config(config: dict[str, Any]) -> OptimizerConfig:
    return OptimizerConfig(
        learning_rate=_float_value(
            config,
            "training.optimizer.learning_rate",
            "optimizer.learning_rate",
            "training.learning_rate",
            default=1e-6,
        ),
        weight_decay=_float_value(
            config,
            "training.optimizer.weight_decay",
            "optimizer.weight_decay",
            "training.weight_decay",
            default=0.0,
        ),
        warmup_steps=_int_value(
            config,
            "training.optimizer.warmup_steps",
            "optimizer.warmup_steps",
            "training.warmup_steps",
            default=0,
        ),
        gradient_accumulation_steps=_int_value(
            config,
            "training.optimizer.gradient_accumulation_steps",
            "optimizer.gradient_accumulation_steps",
            "training.gradient_accumulation_steps",
            default=1,
        ),
        max_grad_norm=_float_value(
            config,
            "training.optimizer.max_grad_norm",
            "optimizer.max_grad_norm",
            "training.max_grad_norm",
            default=1.0,
        ),
    )


def _build_grpo_model_training_config(config: dict[str, Any]) -> GRPOModelTrainingConfig:
    return GRPOModelTrainingConfig(
        optimizer=_build_optimizer_config(config),
        loss=GRPOLossConfig(
            clip_epsilon=_float_value(
                config,
                "training.loss.clip_epsilon",
                "loss.clip_epsilon",
                "training.clip_epsilon",
                default=0.2,
            ),
            kl_beta=_float_value(
                config,
                "training.loss.kl_beta",
                "loss.kl_beta",
                "training.kl_beta",
                default=0.0,
            ),
            max_log_ratio=_float_value(
                config,
                "training.loss.max_log_ratio",
                "loss.max_log_ratio",
                default=20.0,
            ),
        ),
        max_batches=_optional_int_value(
            _first_value(config, "training.max_batches", default=None)
        ),
        step_scheduler=_bool_value(config, "training.step_scheduler", default=True),
        pad_token_id=_int_value(config, "training.pad_token_id", default=0),
        device=_optional_str_value(_first_value(config, "training.device", default=None)),
        dtype=_str_value(config, "training.dtype", "model.dtype", default="fp32"),
        microbatch_size=_int_value(config, "training.microbatch_size", default=0),
    )


def _build_ppo_model_training_config(config: dict[str, Any]) -> PPOModelTrainingConfig:
    value_clip_epsilon = _first_value(
        config,
        "training.loss.value_clip_epsilon",
        "loss.value_clip_epsilon",
        "training.value_clip_epsilon",
        default=0.2,
    )
    return PPOModelTrainingConfig(
        optimizer=_build_optimizer_config(config),
        loss=PPOLossConfig(
            clip_epsilon=_float_value(
                config,
                "training.loss.clip_epsilon",
                "loss.clip_epsilon",
                "training.clip_epsilon",
                default=0.2,
            ),
            value_clip_epsilon=_optional_float_value(value_clip_epsilon),
            value_loss_coef=_float_value(
                config,
                "training.loss.value_loss_coef",
                "loss.value_loss_coef",
                "training.value_loss_coef",
                default=0.5,
            ),
            entropy_coef=_float_value(
                config,
                "training.loss.entropy_coef",
                "loss.entropy_coef",
                "training.entropy_coef",
                default=0.0,
            ),
            kl_beta=_float_value(
                config,
                "training.loss.kl_beta",
                "loss.kl_beta",
                "training.kl_beta",
                default=0.0,
            ),
            gamma=_float_value(
                config,
                "training.loss.gamma",
                "loss.gamma",
                "training.gamma",
                default=1.0,
            ),
            gae_lambda=_float_value(
                config,
                "training.loss.gae_lambda",
                "loss.gae_lambda",
                "training.gae_lambda",
                default=1.0,
            ),
            normalize_advantages=_bool_value(
                config,
                "training.loss.normalize_advantages",
                "loss.normalize_advantages",
                "training.normalize_advantages",
                default=True,
            ),
            max_log_ratio=_float_value(
                config,
                "training.loss.max_log_ratio",
                "loss.max_log_ratio",
                default=20.0,
            ),
        ),
        max_batches=_optional_int_value(
            _first_value(config, "training.max_batches", default=None)
        ),
        step_scheduler=_bool_value(config, "training.step_scheduler", default=True),
        pad_token_id=_int_value(config, "training.pad_token_id", default=0),
        device=_optional_str_value(_first_value(config, "training.device", default=None)),
        dtype=_str_value(config, "training.dtype", "model.dtype", default="fp32"),
    )


def _build_grpo_batch_config(config: dict[str, Any]) -> GRPORolloutBatchConfig:
    return GRPORolloutBatchConfig(
        max_prompt_tokens=_optional_int_value(
            _first_value(config, "batch.max_prompt_tokens", default=None)
        ),
        max_response_tokens=_optional_int_value(
            _first_value(config, "batch.max_response_tokens", default=None)
        ),
        response_source=_str_value(config, "batch.response_source", default="raw_completion"),
        include_failed_parses=_bool_value(
            config,
            "batch.include_failed_parses",
            default=True,
        ),
        min_reward_std=_float_value(config, "batch.min_reward_std", default=1e-8),
        prompt_truncation_side=_str_value(
            config,
            "batch.prompt_truncation_side",
            default="left",
        ),
        response_truncation_side=_str_value(
            config,
            "batch.response_truncation_side",
            default="right",
        ),
    )


def _build_ppo_batch_config(config: dict[str, Any]) -> PPORolloutBatchConfig:
    return PPORolloutBatchConfig(
        max_prompt_tokens=_optional_int_value(
            _first_value(config, "batch.max_prompt_tokens", default=None)
        ),
        max_response_tokens=_optional_int_value(
            _first_value(config, "batch.max_response_tokens", default=None)
        ),
        response_source=_str_value(config, "batch.response_source", default="raw_completion"),
        include_failed_parses=_bool_value(
            config,
            "batch.include_failed_parses",
            default=True,
        ),
        prompt_truncation_side=_str_value(
            config,
            "batch.prompt_truncation_side",
            default="left",
        ),
        response_truncation_side=_str_value(
            config,
            "batch.response_truncation_side",
            default="right",
        ),
    )


def _build_grpo_evaluation_config(
    config: dict[str, Any],
) -> GRPOOnlineEvaluationConfig | None:
    if _first_value(config, "evaluation", default=_MISSING) is _MISSING:
        return None
    if not _bool_value(config, "evaluation.enabled", default=True):
        return None
    return GRPOOnlineEvaluationConfig(
        samples_per_task=_int_value(config, "evaluation.samples_per_task", default=1),
        max_new_tokens=_optional_int_value(
            _first_value(
                config,
                "evaluation.max_new_tokens",
                "evaluation.max_response_tokens",
                default=None,
            )
        ),
        temperature=_float_value(config, "evaluation.temperature", default=0.0),
        top_p=_float_value(config, "evaluation.top_p", default=1.0),
        include_hidden=_bool_value(config, "evaluation.include_hidden", default=True),
        ks=_int_tuple_value(_first_value(config, "evaluation.ks", default=(1,))),
    )


def _build_ppo_evaluation_config(
    config: dict[str, Any],
) -> PPOOnlineEvaluationConfig | None:
    if _first_value(config, "evaluation", default=_MISSING) is _MISSING:
        return None
    if not _bool_value(config, "evaluation.enabled", default=True):
        return None
    return PPOOnlineEvaluationConfig(
        samples_per_task=_int_value(config, "evaluation.samples_per_task", default=1),
        max_new_tokens=_optional_int_value(
            _first_value(
                config,
                "evaluation.max_new_tokens",
                "evaluation.max_response_tokens",
                default=None,
            )
        ),
        temperature=_float_value(config, "evaluation.temperature", default=0.0),
        top_p=_float_value(config, "evaluation.top_p", default=1.0),
        include_hidden=_bool_value(config, "evaluation.include_hidden", default=True),
        ks=_int_tuple_value(_first_value(config, "evaluation.ks", default=(1,))),
    )


def _first_value(config: dict[str, Any], *paths: str, default: Any) -> Any:
    for path in paths:
        value = config_get(config, path, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _int_value(config: dict[str, Any], *paths: str, default: int) -> int:
    return int(_first_value(config, *paths, default=default))


def _float_value(config: dict[str, Any], *paths: str, default: float) -> float:
    return float(_first_value(config, *paths, default=default))


def _str_value(config: dict[str, Any], *paths: str, default: str) -> str:
    return str(_first_value(config, *paths, default=default))


def _bool_value(config: dict[str, Any], *paths: str, default: bool) -> bool:
    return _as_bool(_first_value(config, *paths, default=default))


def _optional_int_value(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_str_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_tuple_value(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        values: Iterable[Any] = (item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, Iterable):
        values = value
    else:
        values = (value,)
    return tuple(int(item) for item in values)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)
