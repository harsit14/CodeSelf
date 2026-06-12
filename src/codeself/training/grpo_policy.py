"""Policy-model helpers for GRPO rollout generation."""

from __future__ import annotations

from typing import Any

from codeself.agent.generation import (
    CodeGenerator,
    GenerationRequest as AgentGenerationRequest,
    GenerationResult,
)
from codeself.training.common import GenerationRequest as EngineGenerationRequest
from codeself.training.common import ModelEngine


class ModelEngineCodeGenerator(CodeGenerator):
    """Adapt a training `ModelEngine` to the agent rollout generator surface."""

    backend_name = "model_engine"

    def __init__(
        self,
        engine: ModelEngine,
        *,
        model_name: str | None = None,
        backend_name: str = "model_engine",
    ) -> None:
        self.engine = engine
        self.model_name = model_name or engine.__class__.__name__
        self.backend_name = backend_name

    def generate(self, request: AgentGenerationRequest) -> GenerationResult:
        sequence = self.engine.generate(
            EngineGenerationRequest(
                prompt=request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                seed=request.seed,
            )
        )
        return self._result_from_sequence(sequence)

    @property
    def supports_batch_generation(self) -> bool:
        return hasattr(self.engine, "generate_batch")

    def generate_batch(
        self,
        requests: list[AgentGenerationRequest],
    ) -> list[GenerationResult]:
        """Generate many completions in one batched engine forward pass."""

        engine_batch = getattr(self.engine, "generate_batch", None)
        if engine_batch is None:
            return [self.generate(request) for request in requests]
        sequences = engine_batch(
            [
                EngineGenerationRequest(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    seed=request.seed,
                )
                for request in requests
            ]
        )
        return [self._result_from_sequence(sequence) for sequence in sequences]

    def _result_from_sequence(self, sequence: Any) -> GenerationResult:
        return GenerationResult(
            text=sequence.response,
            backend=self.backend_name,
            model_name=self.model_name,
            metadata={
                "finish_reason": sequence.finish_reason,
                "prompt_tokens": sequence.prompt_token_count,
                "response_tokens": sequence.response_token_count,
                **_scalar_metadata(sequence.metadata),
            },
        )


def _scalar_metadata(source: dict[str, Any]) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key, value in source.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }
