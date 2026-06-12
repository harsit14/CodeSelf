"""Model engine contracts and optional Transformers adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, TypeAlias

from codeself.training.common.config import ModelRuntimeConfig
from codeself.training.common.logprobs import TokenLogprobs
from codeself.training.common.masking import TokenMask, build_prompt_response_mask
from codeself.training.common.records import SequenceTrainingSample
from codeself.training.common.tokenization import (
    TokenizerEngine,
    encode_prompt_response,
    truncate_token_ids,
)

FinishReason: TypeAlias = Literal["stop", "length", "error", "unknown"]


@dataclass(frozen=True)
class GenerationRequest:
    """Sampling request shared by local and external model engines."""

    prompt: str
    max_new_tokens: int
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class GeneratedSequence:
    """One generated response packed with token masks for training."""

    prompt: str
    response: str
    input_ids: tuple[int, ...]
    masks: TokenMask
    finish_reason: FinishReason = "unknown"
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.input_ids) != self.masks.token_count:
            raise ValueError("input_ids length must match masks")
        if self.finish_reason not in {"stop", "length", "error", "unknown"}:
            raise ValueError("finish_reason must be stop, length, error, or unknown")

    @property
    def prompt_token_count(self) -> int:
        return self.masks.prompt_tokens

    @property
    def response_token_count(self) -> int:
        return self.masks.response_tokens

    def to_training_sample(
        self,
        *,
        task_id: str,
        sample_index: int,
        reward: float,
        logprobs: TokenLogprobs | None = None,
        advantage: float | None = None,
    ) -> SequenceTrainingSample:
        metadata = dict(self.metadata)
        metadata["finish_reason"] = self.finish_reason
        return SequenceTrainingSample(
            task_id=task_id,
            sample_index=sample_index,
            input_ids=self.input_ids,
            masks=self.masks,
            reward=reward,
            logprobs=logprobs,
            advantage=advantage,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "input_ids": list(self.input_ids),
            "masks": self.masks.to_dict(),
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }


class ModelEngine(Protocol):
    """Minimal model surface consumed by future GRPO/PPO trainers."""

    @property
    def tokenizer(self) -> TokenizerEngine:
        """Tokenizer paired with this model engine."""
        ...

    def generate(self, request: GenerationRequest) -> GeneratedSequence:
        """Generate one response for a prompt."""
        ...

    def token_logprobs(self, input_ids: Sequence[int]) -> TokenLogprobs:
        """Return token-aligned policy logprobs for a packed sequence."""
        ...


@dataclass(frozen=True)
class TransformersEngineConfig:
    """Optional Hugging Face model-engine settings."""

    model: ModelRuntimeConfig
    local_files_only: bool = True
    device_map: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model.to_dict(),
            "local_files_only": self.local_files_only,
            "device_map": self.device_map,
        }


class TransformersTokenizerEngine:
    """TokenizerEngine adapter for a Hugging Face tokenizer instance."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @property
    def eos_token_id(self) -> int | None:
        value = getattr(self._tokenizer, "eos_token_id", None)
        return None if value is None else int(value)

    def encode(
        self,
        text: str,
        *,
        max_tokens: int | None = None,
        truncation_side: str = "right",
    ) -> tuple[int, ...]:
        token_ids = tuple(
            int(token_id)
            for token_id in self._tokenizer.encode(text, add_special_tokens=False)
        )
        return truncate_token_ids(
            token_ids,
            max_tokens=max_tokens,
            truncation_side=truncation_side,
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self._tokenizer.decode(
                [int(token_id) for token_id in token_ids],
                skip_special_tokens=True,
            )
        )


class TransformersModelEngine:
    """Lazy optional Hugging Face causal-LM engine."""

    def __init__(self, config: TransformersEngineConfig) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TransformersModelEngine requires optional packages: torch and transformers"
            ) from exc

        self.config = config
        self._torch = torch
        tokenizer_revision = config.model.tokenizer_revision or config.model.revision
        tokenizer_kwargs: dict[str, object] = {
            "trust_remote_code": config.model.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        if tokenizer_revision is not None:
            tokenizer_kwargs["revision"] = tokenizer_revision
        self._raw_tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer_name,
            **tokenizer_kwargs,
        )
        self._tokenizer = TransformersTokenizerEngine(self._raw_tokenizer)
        model_kwargs: dict[str, object] = {
            "trust_remote_code": config.model.trust_remote_code,
            "local_files_only": config.local_files_only,
        }
        if config.model.revision is not None:
            model_kwargs["revision"] = config.model.revision
        torch_dtype = _torch_dtype(torch, config.model.dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if config.device_map is not None:
            model_kwargs["device_map"] = config.device_map
        self._model = AutoModelForCausalLM.from_pretrained(config.model.name, **model_kwargs)
        if config.model.use_lora:
            self._model = attach_lora_adapter(self._model, config.model)
        _maybe_enable_gradient_checkpointing(
            self._model,
            enabled=config.model.gradient_checkpointing,
        )
        self._model.eval()
        if config.device_map is None and config.model.device not in {"auto", "cpu"}:
            self._model.to(config.model.device)
        if config.device_map is None and config.model.device == "cpu":
            self._model.to("cpu")

    @property
    def tokenizer(self) -> TokenizerEngine:
        return self._tokenizer

    @property
    def model(self) -> Any:
        """Return the underlying causal-LM module for trainable GRPO paths."""

        return self._model

    @property
    def raw_tokenizer(self) -> Any:
        """Return the underlying tokenizer for advanced integration paths."""

        return self._raw_tokenizer

    def generate(self, request: GenerationRequest) -> GeneratedSequence:
        if request.seed is not None:
            self._torch.manual_seed(request.seed)
        encoded = self._raw_tokenizer(
            request.prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoded = self._move_batch_to_model_device(encoded)
        prompt_token_count = int(encoded["input_ids"].shape[-1])
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
        }
        if request.temperature > 0:
            generate_kwargs["temperature"] = request.temperature
            generate_kwargs["top_p"] = request.top_p
        if self._raw_tokenizer.eos_token_id is not None:
            generate_kwargs["pad_token_id"] = self._raw_tokenizer.eos_token_id
        with self._torch.no_grad():
            output_ids = self._model.generate(**encoded, **generate_kwargs)
        input_ids = tuple(int(token_id) for token_id in output_ids[0].tolist())
        response_ids = input_ids[prompt_token_count:]
        response = self._tokenizer.decode(response_ids)
        masks = build_prompt_response_mask(
            token_count=len(input_ids),
            prompt_token_count=prompt_token_count,
            response_token_count=len(response_ids),
        )
        finish_reason = "length" if len(response_ids) >= request.max_new_tokens else "stop"
        return GeneratedSequence(
            prompt=request.prompt,
            response=response,
            input_ids=input_ids,
            masks=masks,
            finish_reason=finish_reason,
            metadata={"engine": "transformers"},
        )

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GeneratedSequence]:
        """Generate responses for many prompts in one padded forward pass.

        Requests are bucketed by ``max_new_tokens`` and each bucket is run as a
        single left-padded batch, which is far faster than per-sample calls on
        MPS/GPU. Each row samples independently, giving the within-group
        diversity GRPO relies on. Results are returned in input order.
        """

        request_list = list(requests)
        if not request_list:
            return []
        results: list[GeneratedSequence | None] = [None] * len(request_list)
        buckets: dict[tuple[int, float, float, bool], list[int]] = {}
        for index, request in enumerate(request_list):
            key = (
                request.max_new_tokens,
                request.temperature,
                request.top_p,
                request.temperature > 0,
            )
            buckets.setdefault(key, []).append(index)
        for (max_new_tokens, temperature, top_p, do_sample), indices in buckets.items():
            seed = next(
                (request_list[i].seed for i in indices if request_list[i].seed is not None),
                None,
            )
            if seed is not None:
                self._torch.manual_seed(seed)
            prompts = [request_list[i].prompt for i in indices]
            sequences = self._generate_padded_batch(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            for position, sequence in zip(indices, sequences, strict=True):
                results[position] = sequence
        return [sequence for sequence in results if sequence is not None]

    def _generate_padded_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> list[GeneratedSequence]:
        torch = self._torch
        previous_side = self._raw_tokenizer.padding_side
        self._raw_tokenizer.padding_side = "left"
        pad_token_id = self._raw_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._raw_tokenizer.eos_token_id
        try:
            encoded = self._raw_tokenizer(
                prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
        finally:
            self._raw_tokenizer.padding_side = previous_side
        encoded = self._move_batch_to_model_device(encoded)
        padded_prompt_len = int(encoded["input_ids"].shape[-1])
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        with torch.no_grad():
            output_ids = self._model.generate(**encoded, **generate_kwargs)
        attention = encoded["attention_mask"]
        sequences: list[GeneratedSequence] = []
        for row in range(output_ids.shape[0]):
            prompt_len = int(attention[row].sum().item())
            prompt_ids = tuple(
                int(token_id)
                for token_id in encoded["input_ids"][row, padded_prompt_len - prompt_len :].tolist()
            )
            response_ids = tuple(
                int(token_id) for token_id in output_ids[row, padded_prompt_len:].tolist()
            )
            response_ids = _strip_trailing_pad(response_ids, pad_token_id)
            input_ids = prompt_ids + response_ids
            response = self._tokenizer.decode(response_ids)
            masks = build_prompt_response_mask(
                token_count=len(input_ids),
                prompt_token_count=len(prompt_ids),
                response_token_count=len(response_ids),
            )
            finish_reason = "length" if len(response_ids) >= max_new_tokens else "stop"
            sequences.append(
                GeneratedSequence(
                    prompt=prompts[row],
                    response=response,
                    input_ids=input_ids,
                    masks=masks,
                    finish_reason=finish_reason,
                    metadata={"engine": "transformers", "batched": True},
                )
            )
        return sequences

    def token_logprobs(self, input_ids: Sequence[int]) -> TokenLogprobs:
        ids = tuple(int(token_id) for token_id in input_ids)
        if not ids:
            raise ValueError("input_ids cannot be empty")
        tensor = self._torch.tensor([ids], dtype=self._torch.long)
        tensor = self._move_tensor_to_model_device(tensor)
        with self._torch.no_grad():
            logits = self._model(input_ids=tensor).logits
            shifted_logits = logits[:, :-1, :]
            shifted_labels = tensor[:, 1:]
            log_probs = self._torch.nn.functional.log_softmax(shifted_logits, dim=-1)
            gathered = log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)
        policy_logprobs = (0.0,) + tuple(float(value) for value in gathered[0].tolist())
        return TokenLogprobs(token_ids=ids, policy_logprobs=policy_logprobs)

    def _move_batch_to_model_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self._model_device()
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def _move_tensor_to_model_device(self, tensor: Any) -> Any:
        return tensor.to(self._model_device())

    def _model_device(self) -> Any:
        return next(self._model.parameters()).device


def build_generated_sequence(
    tokenizer: TokenizerEngine,
    *,
    prompt: str,
    response: str,
    max_prompt_tokens: int | None = None,
    max_response_tokens: int | None = None,
    finish_reason: FinishReason = "unknown",
    metadata: dict[str, str | int | float | bool] | None = None,
) -> GeneratedSequence:
    """Pack an already generated response into a training-ready sequence."""

    packed = encode_prompt_response(
        tokenizer,
        prompt=prompt,
        response=response,
        max_prompt_tokens=max_prompt_tokens,
        max_response_tokens=max_response_tokens,
    )
    sequence_metadata = dict(metadata or {})
    sequence_metadata["prompt_truncated"] = packed.prompt.truncated
    sequence_metadata["response_truncated"] = packed.response.truncated
    return GeneratedSequence(
        prompt=prompt,
        response=response,
        input_ids=packed.input_ids,
        masks=packed.masks,
        finish_reason=finish_reason,
        metadata=sequence_metadata,
    )


def _strip_trailing_pad(token_ids: tuple[int, ...], pad_token_id: int | None) -> tuple[int, ...]:
    if pad_token_id is None:
        return token_ids
    end = len(token_ids)
    while end > 0 and token_ids[end - 1] == pad_token_id:
        end -= 1
    return token_ids[:end]


def _torch_dtype(torch: Any, dtype: str) -> Any:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return None


def attach_lora_adapter(model: Any, config: ModelRuntimeConfig) -> Any:
    """Attach a PEFT LoRA adapter to a causal-LM model."""

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "model.use_lora=true requires the optional package: peft"
        ) from exc

    task_type = getattr(TaskType, "CAUSAL_LM", "CAUSAL_LM")
    lora_kwargs: dict[str, object] = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "task_type": task_type,
    }
    if config.lora_target_modules:
        lora_kwargs["target_modules"] = list(config.lora_target_modules)
    return get_peft_model(model, LoraConfig(**lora_kwargs))


def _maybe_enable_gradient_checkpointing(model: Any, *, enabled: bool) -> None:
    if not enabled:
        return
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "use_cache"):
        model_config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
