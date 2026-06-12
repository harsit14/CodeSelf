"""Reward functions for execution feedback."""

from codeself.rewards.composite import CompositeRewardScorer
from codeself.rewards.correctness import (
    AppliedPenalty,
    RewardBreakdown,
    RewardComponent,
    RewardConfig,
    RewardPenalties,
    RewardWeights,
    score_correctness,
)
from codeself.rewards.efficiency import EfficiencyMetrics, measure_efficiency
from codeself.rewards.modes import (
    ConfigurableRewardScorer,
    RewardModeConfig,
    RewardScorer,
    make_reward_scorer,
    make_reward_scorer_from_mode,
)
from codeself.rewards.quality import QualityMetrics, measure_quality
from codeself.rewards.reward_hacking import RewardHackingReport, detect_reward_hacking

__all__ = [
    "AppliedPenalty",
    "CompositeRewardScorer",
    "EfficiencyMetrics",
    "QualityMetrics",
    "RewardHackingReport",
    "RewardBreakdown",
    "RewardComponent",
    "RewardConfig",
    "RewardModeConfig",
    "RewardPenalties",
    "RewardScorer",
    "RewardWeights",
    "ConfigurableRewardScorer",
    "detect_reward_hacking",
    "make_reward_scorer",
    "make_reward_scorer_from_mode",
    "measure_efficiency",
    "measure_quality",
    "score_correctness",
]
