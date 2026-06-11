"""Dependency-free smoke trainers.

These modules preserve the original CI-friendly training diagnostics. They do
not update model weights; real GRPO/PPO implementations should live beside this
package and share the rollout/reward contracts.
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
from codeself.training.smoke.ppo import (
    PPOSmokeConfig,
    PPOSmokeResult,
    PPOSmokeTrainer,
    PPOStepMetrics,
    summarize_ppo_step,
)

__all__ = [
    "GRPOSmokeConfig",
    "GRPOSmokeResult",
    "GRPOSmokeTrainer",
    "GRPOStepMetrics",
    "GroupAdvantage",
    "PPOSmokeConfig",
    "PPOSmokeResult",
    "PPOSmokeTrainer",
    "PPOStepMetrics",
    "compute_group_advantages",
    "summarize_grpo_step",
    "summarize_ppo_step",
]
