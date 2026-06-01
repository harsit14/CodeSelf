"""Generation backend abstractions."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GenerationRequest:
    """One generation request."""

    task_id: str
    prompt: str
    sample_index: int
    seed: int
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    entry_point: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Raw completion from a generation backend."""

    text: str
    backend: str
    model_name: str
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)


class CodeGenerator(ABC):
    """Abstract generation backend."""

    backend_name: str
    model_name: str

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one completion."""


class MockGenerator(CodeGenerator):
    """Deterministic local generator for pipeline tests."""

    backend_name = "mock"
    model_name = "mock-deterministic-v1"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        entry_point = request.entry_point or "solution"
        bucket = _stable_bucket(request.task_id, request.sample_index, request.seed)
        if bucket == 0:
            body = f"def {entry_point}(*args, **kwargs):\n    return None\n"
        else:
            body = f"def {entry_point}(x):\n    return x + 1\n"
        completion = f"```python\n{body}```"
        return GenerationResult(
            text=completion,
            backend=self.backend_name,
            model_name=self.model_name,
            metadata={"bucket": bucket},
        )


class StaticGenerator(CodeGenerator):
    """Generator that always returns the same completion."""

    backend_name = "static"

    def __init__(self, completion: str, *, model_name: str = "static-v1") -> None:
        self.completion = completion
        self.model_name = model_name

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text=self.completion,
            backend=self.backend_name,
            model_name=self.model_name,
            metadata={"sample_index": request.sample_index},
        )


class TransformersGenerator(CodeGenerator):
    """Optional local Transformers backend.

    This backend is intentionally lazy so the project can be tested without
    installing model dependencies or downloading weights.
    """

    backend_name = "transformers"

    def __init__(self, model_name_or_path: str, *, device: str | None = None) -> None:
        self.model_name = model_name_or_path
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Install the training extra or use --backend mock."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, local_files_only=True)
        if device:
            self.model.to(device)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        inputs = self.tokenizer(request.prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        output = self.model.generate(
            **inputs,
            do_sample=request.temperature > 0,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            backend=self.backend_name,
            model_name=self.model_name,
            metadata={
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
        )


def make_generator(
    backend: str,
    *,
    model_name_or_path: str | None = None,
    static_completion: str | None = None,
) -> CodeGenerator:
    """Factory for generation backends."""

    if backend == "mock":
        return MockGenerator()
    if backend == "static":
        if static_completion is None:
            raise ValueError("static backend requires static_completion")
        return StaticGenerator(static_completion)
    if backend == "transformers":
        if not model_name_or_path:
            raise ValueError("transformers backend requires model_name_or_path")
        return TransformersGenerator(model_name_or_path)
    raise ValueError(f"unknown generation backend: {backend}")


def _stable_bucket(task_id: str, sample_index: int, seed: int) -> int:
    payload = f"{task_id}:{sample_index}:{seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % 3
