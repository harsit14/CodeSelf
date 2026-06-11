"""Compatibility wrapper for dependency-free PPO smoke diagnostics.

The original smoke trainer now lives under :mod:`codeself.training.smoke.ppo`
so the real PPO baseline can grow beside it while old scripts and imports keep
working.
"""

from codeself.training.smoke.ppo import (
    PPOSmokeConfig,
    PPOSmokeResult,
    PPOSmokeTrainer,
    PPOStepMetrics,
    summarize_ppo_step,
)

__all__ = [
    "PPOSmokeConfig",
    "PPOSmokeResult",
    "PPOSmokeTrainer",
    "PPOStepMetrics",
    "summarize_ppo_step",
]
