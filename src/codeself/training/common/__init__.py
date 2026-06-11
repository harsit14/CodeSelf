"""Shared contracts for real training backends.

The common package is dependency-free by design. It defines the data shapes and
selection logic that GRPO, PPO, and future TRL/verl integrations should share
without importing heavyweight model libraries during smoke tests.
"""

from codeself.training.common.backends import (
    BackendAvailability,
    BackendSpec,
    TrainerBackend,
    backend_availability,
    get_backend_spec,
    list_backend_specs,
)
from codeself.training.common.config import (
    ModelRuntimeConfig,
    OptimizerConfig,
    RolloutRuntimeConfig,
    TrainingCoreConfig,
    build_training_core_config,
)
from codeself.training.common.engine import (
    GeneratedSequence,
    GenerationRequest,
    ModelEngine,
    TransformersEngineConfig,
    TransformersModelEngine,
    TransformersTokenizerEngine,
    build_generated_sequence,
)
from codeself.training.common.logprobs import (
    LogprobSummary,
    TokenLogprobs,
    summarize_logprobs,
)
from codeself.training.common.masking import (
    TokenMask,
    build_prompt_response_mask,
    mask_mean,
    mask_sum,
)
from codeself.training.common.records import (
    SequenceTrainingSample,
    TrainingBatch,
)
from codeself.training.common.tokenization import (
    TokenizedPromptResponse,
    TokenizedText,
    TokenizerEngine,
    WhitespaceTokenizerEngine,
    encode_prompt_response,
    encode_text,
    truncate_token_ids,
)

__all__ = [
    "BackendAvailability",
    "BackendSpec",
    "GeneratedSequence",
    "GenerationRequest",
    "LogprobSummary",
    "ModelEngine",
    "ModelRuntimeConfig",
    "OptimizerConfig",
    "RolloutRuntimeConfig",
    "SequenceTrainingSample",
    "TokenizedPromptResponse",
    "TokenizedText",
    "TokenizerEngine",
    "TokenLogprobs",
    "TokenMask",
    "TrainerBackend",
    "TransformersEngineConfig",
    "TransformersModelEngine",
    "TransformersTokenizerEngine",
    "TrainingBatch",
    "TrainingCoreConfig",
    "WhitespaceTokenizerEngine",
    "backend_availability",
    "build_generated_sequence",
    "build_prompt_response_mask",
    "build_training_core_config",
    "encode_prompt_response",
    "encode_text",
    "get_backend_spec",
    "list_backend_specs",
    "mask_mean",
    "mask_sum",
    "summarize_logprobs",
    "truncate_token_ids",
]
