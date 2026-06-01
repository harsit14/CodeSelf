"""Training entry points for GRPO/PPO experiments."""

from codeself.training.algorithm_compare import (
    AlgorithmComparison,
    compare_algorithm_metrics,
    read_last_metric,
    write_algorithm_comparison,
)
from codeself.training.callbacks import JsonlMetricWriter, write_checkpoint_manifest
from codeself.training.experiments import (
    ExperimentRunSummary,
    ExperimentVariant,
    aggregate_by_variant_name,
    build_grpo_experiment_matrix,
    select_best_run,
    write_experiment_summary_json,
    write_experiment_table,
)
from codeself.training.grpo_train import (
    GRPOSmokeConfig,
    GRPOSmokeResult,
    GRPOSmokeTrainer,
    GRPOStepMetrics,
    GroupAdvantage,
    compute_group_advantages,
    summarize_grpo_step,
)
from codeself.training.ppo_train import (
    PPOSmokeConfig,
    PPOSmokeResult,
    PPOSmokeTrainer,
    PPOStepMetrics,
    summarize_ppo_step,
)

__all__ = [
    "AlgorithmComparison",
    "GRPOSmokeConfig",
    "GRPOSmokeResult",
    "GRPOSmokeTrainer",
    "GRPOStepMetrics",
    "GroupAdvantage",
    "JsonlMetricWriter",
    "PPOSmokeConfig",
    "PPOSmokeResult",
    "PPOSmokeTrainer",
    "PPOStepMetrics",
    "ExperimentRunSummary",
    "ExperimentVariant",
    "aggregate_by_variant_name",
    "build_grpo_experiment_matrix",
    "compare_algorithm_metrics",
    "compute_group_advantages",
    "read_last_metric",
    "select_best_run",
    "summarize_grpo_step",
    "summarize_ppo_step",
    "write_algorithm_comparison",
    "write_checkpoint_manifest",
    "write_experiment_summary_json",
    "write_experiment_table",
]
