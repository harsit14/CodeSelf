"""Replay an online-training artifact directory into an experiment tracker.

Online GRPO/PPO runs already write per-cycle ``metrics.jsonl``,
``evaluation.json``, and ``rollouts.jsonl``. Rather than couple the training
loop to a tracker, this module reuses the learning-curve parser to push those
cycle summaries into any ``ExperimentTracker`` after (or during) a run.
"""

from __future__ import annotations

from pathlib import Path

from codeself.evaluation.learning_curves import collect_learning_curve
from codeself.tracking.trackers import ExperimentTracker


def log_learning_curve_to_tracker(
    artifact_dir: str | Path,
    tracker: ExperimentTracker,
    *,
    label: str = "run",
) -> int:
    """Push every cycle's metrics from an artifact dir into a tracker.

    Returns the number of cycles logged. Each cycle becomes one tracker step
    with train/eval pass@1, reward, response length, degenerate rate, and the
    available optimization diagnostics (loss, KL, value/entropy loss).
    """

    run = collect_learning_curve(label, artifact_dir)
    for point in run.points:
        metrics: dict[str, float] = {
            "train/pass_at_1": point.train_pass_at_1,
            "train/reward_mean": point.train_reward_mean,
            "train/degenerate_rate": point.train_degenerate_rate,
            "train/response_token_mean": point.train_response_token_mean,
            "train/rollout_count": float(point.rollout_count),
            "train/optimizer_steps": float(point.optimizer_steps),
            "loss/total": point.train_loss,
            "loss/policy": point.policy_loss,
            "loss/value": point.value_loss,
            "loss/entropy": point.entropy_loss,
            "loss/kl": point.kl_loss,
            "kl/mean_approx": point.mean_approx_kl,
        }
        if point.eval_pass_at_1 is not None:
            metrics["eval/pass_at_1"] = point.eval_pass_at_1
        if point.eval_reward_mean is not None:
            metrics["eval/reward_mean"] = point.eval_reward_mean
        if point.eval_degenerate_rate is not None:
            metrics["eval/degenerate_rate"] = point.eval_degenerate_rate
        if point.eval_response_token_mean is not None:
            metrics["eval/response_token_mean"] = point.eval_response_token_mean
        tracker.log(metrics, step=point.cycle)
    tracker.finish()
    return run.cycle_count
