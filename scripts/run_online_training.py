#!/usr/bin/env python3
"""Run online GRPO/PPO training from one JSON/YAML experiment config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import run_online_training_from_config_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON/simple-YAML config.")
    parser.add_argument("--tasks", type=Path, help="Override config data.tasks.")
    parser.add_argument("--eval-tasks", type=Path, help="Override config data.eval_tasks.")
    parser.add_argument("--output-dir", type=Path, help="Override config output.dir.")
    parser.add_argument("--algorithm", choices=("grpo", "ppo"), help="Override algorithm.")
    args = parser.parse_args()

    launch = run_online_training_from_config_file(
        args.config,
        tasks_path=args.tasks,
        eval_tasks_path=args.eval_tasks,
        artifact_dir=args.output_dir,
        algorithm=args.algorithm,
    )
    summary = launch.summary_dict()
    print(f"wrote online training outputs to {summary['artifact_dir']}")
    for key in (
        "algorithm",
        "model_backend",
        "generation_backend",
        "task_count",
        "eval_task_count",
        "cycles",
        "rollout_mode",
        "total_rollouts",
        "records_used",
        "total_optimizer_steps",
    ):
        print(f"{key}: {summary[key]}")
    print(f"mean_reward: {float(summary['mean_reward']):.4f}")
    print(f"launch_summary: {Path(str(summary['artifact_dir'])) / 'launch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
