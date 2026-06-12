"""Single-config launcher for online GRPO/PPO training."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from codeself.agent import (
    CodeGenerator,
    get_prompt_template,
    make_generator,
)
from codeself.config import config_get, load_config_file
from codeself.datasets import Split, TaskRegistry, TaskSpec, check_dataset_quality
from codeself.rewards import make_reward_scorer_from_mode
from codeself.training.common import (
    ModelRuntimeConfig,
    TransformersEngineConfig,
    TransformersModelEngine,
    truncate_token_ids,
)
from codeself.training.grpo_online import GRPOOnlineTrainingResult, run_grpo_online_training
from codeself.training.grpo_policy import ModelEngineCodeGenerator
from codeself.training.online_config import (
    OnlineAlgorithm,
    OnlineTrainingConfig,
    build_online_training_config,
    resolve_online_algorithm,
)
from codeself.training.ppo_online import PPOOnlineTrainingResult, run_ppo_online_training
from codeself.training.ppo_value_head import CausalLMWithValueHead

OnlineTrainingResult = GRPOOnlineTrainingResult | PPOOnlineTrainingResult


@dataclass(frozen=True)
class OnlineTrainingLaunchResult:
    """Compact receipt for one config-launched online training run."""

    algorithm: OnlineAlgorithm
    config: OnlineTrainingConfig
    result: OnlineTrainingResult
    task_count: int
    eval_task_count: int
    artifact_dir: str
    model_backend: str
    generation_backend: str
    config_path: str | None = None

    @property
    def cycle_count(self) -> int:
        return self.result.cycle_count

    @property
    def total_rollouts(self) -> int:
        return self.result.total_rollouts

    @property
    def records_used(self) -> int:
        return self.result.records_used

    @property
    def total_optimizer_steps(self) -> int:
        return self.result.total_optimizer_steps

    @property
    def mean_reward(self) -> float:
        return self.result.mean_reward

    def summary_dict(self) -> dict[str, object]:
        """Return a compact run summary suitable for `launch_summary.json`."""

        return {
            "algorithm": self.algorithm,
            "config_path": self.config_path,
            "artifact_dir": self.artifact_dir,
            "model_backend": self.model_backend,
            "generation_backend": self.generation_backend,
            "task_count": self.task_count,
            "eval_task_count": self.eval_task_count,
            "cycles": self.config.cycles,
            "cycle_count": self.cycle_count,
            "rollout_mode": self.config.cycle.rollout_mode,
            "samples_per_task": self.config.cycle.rollout.samples_per_task,
            "total_rollouts": self.total_rollouts,
            "records_used": self.records_used,
            "total_optimizer_steps": self.total_optimizer_steps,
            "mean_reward": self.mean_reward,
            "evaluation_enabled": self.config.evaluation is not None,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.summary_dict(),
            "online_config": self.config.to_dict(),
            "online_result": self.result.to_dict(),
        }


def run_online_training_from_config_file(
    path: str | Path,
    *,
    tasks_path: str | Path | None = None,
    eval_tasks_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    algorithm: str | None = None,
) -> OnlineTrainingLaunchResult:
    """Load a JSON/YAML config file and run online GRPO/PPO training."""

    config_path = Path(path)
    config = load_config_file(config_path)
    return run_online_training_from_config(
        config,
        config_path=config_path,
        tasks_path=tasks_path,
        eval_tasks_path=eval_tasks_path,
        artifact_dir=artifact_dir,
        algorithm=algorithm,
    )


def run_online_training_from_config(
    config: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    tasks_path: str | Path | None = None,
    eval_tasks_path: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    algorithm: str | None = None,
) -> OnlineTrainingLaunchResult:
    """Assemble tasks, models, generator, scorer, and run online training."""

    resolved_algorithm = resolve_online_algorithm(config, algorithm)
    online_config = build_online_training_config(config, algorithm=resolved_algorithm)
    selected_tasks = _load_training_tasks(config, tasks_path)
    selected_eval_tasks = _load_eval_tasks(config, eval_tasks_path)
    output_dir = _artifact_dir(config, artifact_dir)
    components = _build_components(config, resolved_algorithm)
    prompt_template = get_prompt_template(
        str(config_get(config, "prompt.template", "direct_solution_v1"))
    )
    eval_prompt_template = get_prompt_template(
        str(config_get(config, "evaluation.prompt_template", prompt_template.name))
    )
    scorer = _build_reward_scorer(config)
    metadata = _run_metadata(config, config_path)

    if resolved_algorithm == "grpo":
        result = run_grpo_online_training(
            selected_tasks,
            generator=components.generator,
            prompt_template=prompt_template,
            tokenizer=components.tokenizer,
            policy_model=components.policy_model,
            old_policy_model=components.old_policy_model,
            reference_model=components.reference_model,
            eval_tasks=selected_eval_tasks,
            eval_generator=components.eval_generator,
            eval_prompt_template=eval_prompt_template,
            scorer=scorer,
            eval_scorer=scorer,
            config=online_config,
            metadata=metadata,
            artifact_dir=output_dir,
        )
    else:
        result = run_ppo_online_training(
            selected_tasks,
            generator=components.generator,
            prompt_template=prompt_template,
            tokenizer=components.tokenizer,
            policy_model=components.policy_model,
            value_model=components.value_model,
            old_policy_model=components.old_policy_model,
            old_value_model=components.old_value_model,
            reference_model=components.reference_model,
            eval_tasks=selected_eval_tasks,
            eval_generator=components.eval_generator,
            eval_prompt_template=eval_prompt_template,
            scorer=scorer,
            eval_scorer=scorer,
            config=online_config,
            metadata=metadata,
            artifact_dir=output_dir,
        )

    launch = OnlineTrainingLaunchResult(
        algorithm=resolved_algorithm,
        config=online_config,
        result=result,
        task_count=len(selected_tasks),
        eval_task_count=len(selected_eval_tasks),
        artifact_dir=str(output_dir),
        model_backend=components.model_backend,
        generation_backend=components.generation_backend,
        config_path=str(config_path) if config_path is not None else None,
    )
    _write_launch_summary(launch)
    return launch


@dataclass(frozen=True)
class _LauncherComponents:
    model_backend: str
    generation_backend: str
    generator: CodeGenerator
    eval_generator: CodeGenerator | None
    tokenizer: Any
    policy_model: Any
    value_model: Any | None = None
    old_policy_model: Any | None = None
    old_value_model: Any | None = None
    reference_model: Any | None = None


def _load_training_tasks(
    config: dict[str, Any],
    tasks_path: str | Path | None,
) -> list[TaskSpec]:
    selected_path = _path_value(tasks_path, config_get(config, "data.tasks"))
    if selected_path is None:
        raise ValueError("data.tasks is required unless --tasks is provided")
    registry = TaskRegistry.from_jsonl(selected_path)
    _check_quality(config, list(registry), "data")
    tasks = _select_tasks(
        registry,
        split=config_get(config, "data.split"),
        limit=config_get(config, "data.limit"),
    )
    if not tasks:
        raise ValueError("no training tasks selected")
    return tasks


def _load_eval_tasks(
    config: dict[str, Any],
    eval_tasks_path: str | Path | None,
) -> list[TaskSpec]:
    selected_path = _path_value(eval_tasks_path, config_get(config, "data.eval_tasks"))
    eval_split = config_get(config, "data.eval_split")
    if selected_path is None and eval_split is None:
        return []
    source_path = selected_path or _path_value(None, config_get(config, "data.tasks"))
    if source_path is None:
        raise ValueError("data.eval_tasks or data.tasks is required for evaluation tasks")
    registry = TaskRegistry.from_jsonl(source_path)
    _check_quality(config, list(registry), "evaluation data")
    tasks = _select_tasks(
        registry,
        split=eval_split,
        limit=config_get(config, "data.eval_limit"),
    )
    if not tasks:
        raise ValueError("no evaluation tasks selected")
    return tasks


def _select_tasks(
    registry: TaskRegistry,
    *,
    split: object,
    limit: object,
) -> list[TaskSpec]:
    tasks = registry.by_split(Split(str(split))) if split is not None else list(registry)
    if limit is not None:
        tasks = tasks[: int(limit)]
    return tasks


def _check_quality(config: dict[str, Any], tasks: list[TaskSpec], label: str) -> None:
    if bool(config_get(config, "data.skip_quality_checks", False)):
        return
    quality = check_dataset_quality(
        tasks,
        near_duplicate_threshold=float(
            config_get(config, "data.near_duplicate_threshold", 0.92)
        ),
    )
    if quality.passed:
        return
    failures = "; ".join(f"{failure.check}: {failure.message}" for failure in quality.failures)
    raise ValueError(f"{label} quality checks failed: {failures}")


def _artifact_dir(config: dict[str, Any], artifact_dir: str | Path | None) -> Path:
    selected = _path_value(
        artifact_dir,
        config_get(config, "output.artifact_dir", config_get(config, "output.dir")),
    )
    if selected is None:
        raise ValueError("output.artifact_dir or output.dir is required")
    return selected


def _build_components(
    config: dict[str, Any],
    algorithm: OnlineAlgorithm,
) -> _LauncherComponents:
    backend = str(config_get(config, "model.backend", "toy")).lower()
    if backend == "toy":
        return _build_toy_components(config)
    if backend == "transformers":
        return _build_transformers_components(config, algorithm)
    raise ValueError("model.backend must be toy or transformers")


def _build_toy_components(config: dict[str, Any]) -> _LauncherComponents:
    vocab_size = int(config_get(config, "model.vocab_size", 2048))
    policy_model = _TinyOnlineCausalLM(vocab_size=vocab_size)
    generator, generation_backend = _build_non_policy_generator(
        config,
        default_backend="mock",
    )
    old_policy_model = (
        _TinyOnlineCausalLM(vocab_size=vocab_size)
        if bool(config_get(config, "model.old_policy.enabled", False))
        else None
    )
    old_value_model = (
        _TinyOnlineCausalLM(vocab_size=vocab_size)
        if bool(config_get(config, "model.old_value.enabled", False))
        else None
    )
    reference_model = (
        _TinyOnlineCausalLM(vocab_size=vocab_size)
        if bool(config_get(config, "model.reference.enabled", False))
        else None
    )
    return _LauncherComponents(
        model_backend="toy",
        generation_backend=generation_backend,
        generator=generator,
        eval_generator=None,
        tokenizer=_SequentialToyTokenizer(eos_token_id=0),
        policy_model=policy_model,
        old_policy_model=old_policy_model,
        old_value_model=old_value_model,
        reference_model=reference_model,
    )


def _build_transformers_components(
    config: dict[str, Any],
    algorithm: OnlineAlgorithm,
) -> _LauncherComponents:
    model_config = _build_model_runtime_config(config)
    engine_config = TransformersEngineConfig(
        model=model_config,
        local_files_only=bool(config_get(config, "model.local_files_only", True)),
        device_map=_optional_str(config_get(config, "model.device_map")),
    )
    policy_engine = TransformersModelEngine(engine_config)
    policy_model = policy_engine.model
    value_model = None
    old_value_model = None
    if algorithm == "ppo":
        policy_model = CausalLMWithValueHead(
            policy_engine.model,
            hidden_size=_optional_int(config_get(config, "model.value_head.hidden_size")),
        )
    generator = ModelEngineCodeGenerator(
        policy_engine,
        model_name=model_config.name,
        backend_name="policy_engine",
    )
    if algorithm == "ppo":
        old_policy_model = _optional_transformers_value_head_model(
            config,
            engine_config,
            "model.old_policy.enabled",
        )
        if bool(config_get(config, "model.old_value.enabled", False)):
            old_value_model = old_policy_model or _optional_transformers_value_head_model(
                config,
                engine_config,
                "model.old_value.enabled",
            )
    else:
        old_policy_model = _optional_transformers_model(
            config,
            engine_config,
            "model.old_policy.enabled",
        )
    reference_model = _optional_transformers_model(
        config,
        _reference_engine_config(config, engine_config),
        "model.reference.enabled",
    )
    return _LauncherComponents(
        model_backend="transformers",
        generation_backend="policy_engine",
        generator=generator,
        eval_generator=None,
        tokenizer=policy_engine.tokenizer,
        policy_model=policy_model,
        value_model=value_model,
        old_policy_model=old_policy_model,
        old_value_model=old_value_model,
        reference_model=reference_model,
    )


def _build_model_runtime_config(config: dict[str, Any]) -> ModelRuntimeConfig:
    model_name = (
        config_get(config, "model.name")
        or config_get(config, "model.model_name_or_path")
        or config_get(config, "generation.model")
    )
    if not model_name:
        raise ValueError("model.name is required for model.backend=transformers")
    use_lora = bool(config_get(config, "model.use_lora", False))
    full_finetune = bool(config_get(config, "model.full_finetune", not use_lora))
    return ModelRuntimeConfig(
        name=str(model_name),
        tokenizer=_optional_str(config_get(config, "model.tokenizer")),
        revision=_optional_str(config_get(config, "model.revision")),
        tokenizer_revision=_optional_str(config_get(config, "model.tokenizer_revision")),
        dtype=str(config_get(config, "model.dtype", "fp32")),
        device=str(config_get(config, "model.device", "auto")),
        trust_remote_code=bool(config_get(config, "model.trust_remote_code", False)),
        gradient_checkpointing=bool(config_get(config, "model.gradient_checkpointing", True)),
        use_lora=use_lora,
        lora_rank=int(config_get(config, "model.lora_rank", 16)),
        lora_alpha=int(config_get(config, "model.lora_alpha", 32)),
        lora_dropout=float(config_get(config, "model.lora_dropout", 0.0)),
        lora_target_modules=_str_tuple(config_get(config, "model.lora_target_modules", ())),
        full_finetune=full_finetune,
    )


def _reference_engine_config(
    config: dict[str, Any],
    engine_config: TransformersEngineConfig,
) -> TransformersEngineConfig:
    if not bool(config_get(config, "model.reference.enabled", False)):
        return engine_config
    reference_use_lora = bool(config_get(config, "model.reference.use_lora", False))
    return replace(
        engine_config,
        model=replace(
            engine_config.model,
            use_lora=reference_use_lora,
            full_finetune=not reference_use_lora,
        ),
    )


def _optional_transformers_model(
    config: dict[str, Any],
    engine_config: TransformersEngineConfig,
    enabled_path: str,
) -> Any | None:
    if not bool(config_get(config, enabled_path, False)):
        return None
    return TransformersModelEngine(engine_config).model


def _optional_transformers_value_head_model(
    config: dict[str, Any],
    engine_config: TransformersEngineConfig,
    enabled_path: str,
) -> CausalLMWithValueHead | None:
    if not bool(config_get(config, enabled_path, False)):
        return None
    return CausalLMWithValueHead(
        TransformersModelEngine(engine_config).model,
        hidden_size=_optional_int(config_get(config, "model.value_head.hidden_size")),
    )


def _build_non_policy_generator(
    config: dict[str, Any],
    *,
    default_backend: str,
) -> tuple[CodeGenerator, str]:
    backend = str(config_get(config, "generation.backend", default_backend))
    if backend in {"policy", "policy_engine", "model_engine"}:
        backend = default_backend
    static_completion = config_get(config, "generation.static_completion")
    static_completion_path = _path_value(
        None,
        config_get(config, "generation.static_completion_file"),
    )
    if static_completion_path is not None:
        static_completion = static_completion_path.read_text(encoding="utf-8")
    generator = make_generator(
        backend,
        model_name_or_path=_optional_str(config_get(config, "generation.model")),
        static_completion=_optional_str(static_completion),
    )
    return generator, backend


def _build_reward_scorer(config: dict[str, Any]) -> Any:
    return make_reward_scorer_from_mode(
        str(config_get(config, "reward.mode", "correctness_v0")),
        compile_success_bonus=float(config_get(config, "reward.compile_success_bonus", 0.0)),
        length_penalty_per_1k_chars=float(
            config_get(config, "reward.length_penalty_per_1k_chars", 0.0)
        ),
    )


def _run_metadata(
    config: dict[str, Any],
    config_path: str | Path | None,
) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "launch_surface": "run_online_training",
    }
    if config_path is not None:
        metadata["config_path"] = str(config_path)
    experiment_name = config_get(config, "experiment.name")
    if experiment_name is not None:
        metadata["experiment_name"] = str(experiment_name)
    return metadata


def _write_launch_summary(launch: OnlineTrainingLaunchResult) -> None:
    output_path = Path(launch.artifact_dir) / "launch_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(launch.summary_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _path_value(cli_value: str | Path | None, config_value: Any) -> Path | None:
    value = cli_value if cli_value is not None else config_value
    if value is None:
        return None
    return Path(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


class _TinyOnlineCausalLM:
    """Tiny trainable causal-LM/value-head surface for launcher tests."""

    def __init__(self, *, vocab_size: int) -> None:
        torch = _require_torch()
        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        self.value_table = torch.nn.Parameter(torch.zeros(vocab_size))

    def parameters(self) -> tuple[Any, Any]:
        return (self.logit_table, self.value_table)

    def train(self) -> None:
        return None

    def eval(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        return {
            "logit_table": self.logit_table.detach().clone(),
            "value_table": self.value_table.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        torch = _require_torch()
        with torch.no_grad():
            self.logit_table.copy_(state["logit_table"])
            self.value_table.copy_(state["value_table"])

    def __call__(self, *, input_ids: Any, attention_mask: Any) -> Any:
        return SimpleNamespace(
            logits=self.logit_table[input_ids],
            values=self.value_table[input_ids],
        )


class _SequentialToyTokenizer:
    """Bounded token-id tokenizer paired with the tiny launcher model."""

    def __init__(self, *, eos_token_id: int | None = None) -> None:
        self._eos_token_id = eos_token_id
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    def encode(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        truncation_side: str = "right",
    ) -> tuple[int, ...]:
        token_ids = tuple(self._id_for_token(token) for token in text.split())
        return truncate_token_ids(
            token_ids,
            max_tokens=max_tokens,
            truncation_side=truncation_side,  # type: ignore[arg-type]
        )

    def decode(self, token_ids: Any) -> str:
        return " ".join(
            self._id_to_token.get(int(token_id), f"<tok:{int(token_id)}>")
            for token_id in token_ids
        )

    def _id_for_token(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = len(self._token_to_id) + 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("online launcher toy backend requires torch") from exc
    return torch
