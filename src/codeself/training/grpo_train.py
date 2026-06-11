"""Compatibility wrapper for dependency-free GRPO smoke diagnostics.

The original smoke trainer now lives under :mod:`codeself.training.smoke.grpo`
so real GRPO code can be added under the training package without blurring the
line between diagnostics and weight-updating training.
"""

from codeself.training.smoke.grpo import (
    GRPOSmokeConfig,
    GRPOSmokeResult,
    GRPOSmokeTrainer,
    GRPOStepMetrics,
    GroupAdvantage,
    compute_group_advantages,
    summarize_grpo_step,
)

__all__ = [
    "GRPOSmokeConfig",
    "GRPOSmokeResult",
    "GRPOSmokeTrainer",
    "GRPOStepMetrics",
    "GroupAdvantage",
    "compute_group_advantages",
    "summarize_grpo_step",
]
