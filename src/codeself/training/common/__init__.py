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

__all__ = [
    "BackendAvailability",
    "BackendSpec",
    "LogprobSummary",
    "ModelRuntimeConfig",
    "OptimizerConfig",
    "RolloutRuntimeConfig",
    "SequenceTrainingSample",
    "TokenLogprobs",
    "TokenMask",
    "TrainerBackend",
    "TrainingBatch",
    "TrainingCoreConfig",
    "backend_availability",
    "build_prompt_response_mask",
    "build_training_core_config",
    "get_backend_spec",
    "list_backend_specs",
    "mask_mean",
    "mask_sum",
    "summarize_logprobs",
]
