"""Evaluation and statistical testing helpers."""

from codeself.evaluation.evaluate import (
    EvaluationSummary,
    GroupEvaluation,
    TaskEvaluation,
    evaluate_rollout_file,
    evaluate_rollouts,
    write_evaluation_report,
)
from codeself.evaluation.pass_at_k import estimate_pass_at_k, mean_pass_at_k
from codeself.evaluation.power_analysis import (
    PowerAnalysisResult,
    approximate_minimum_detectable_effect,
)
from codeself.evaluation.statistical_tests import (
    BootstrapCI,
    ComparisonSummary,
    EffectSizeSummary,
    McNemarResult,
    PairedPermutationResult,
    PairedTaskOutcome,
    bootstrap_delta_ci,
    compare_rollout_files,
    compare_rollouts,
    effect_size_summary,
    exact_mcnemar,
    paired_task_outcomes,
    paired_permutation_test,
    write_comparison_report,
)

__all__ = [
    "BootstrapCI",
    "ComparisonSummary",
    "EffectSizeSummary",
    "EvaluationSummary",
    "GroupEvaluation",
    "McNemarResult",
    "PairedPermutationResult",
    "PairedTaskOutcome",
    "PowerAnalysisResult",
    "TaskEvaluation",
    "approximate_minimum_detectable_effect",
    "bootstrap_delta_ci",
    "compare_rollout_files",
    "compare_rollouts",
    "effect_size_summary",
    "estimate_pass_at_k",
    "exact_mcnemar",
    "evaluate_rollout_file",
    "evaluate_rollouts",
    "mean_pass_at_k",
    "paired_task_outcomes",
    "paired_permutation_test",
    "write_comparison_report",
    "write_evaluation_report",
]
