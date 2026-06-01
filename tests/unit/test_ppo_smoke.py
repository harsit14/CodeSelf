from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import DIRECT_SOLUTION_TEMPLATE, MockGenerator  # noqa: E402
from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402
from codeself.training import (  # noqa: E402
    PPOSmokeConfig,
    PPOSmokeTrainer,
    compare_algorithm_metrics,
    summarize_ppo_step,
)


def _task(task_id: str = "ppo/add-one") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


class PPOSmokeTests(unittest.TestCase):
    def test_ppo_smoke_trainer_runs_and_writes_checkpoint_payload(self) -> None:
        result = PPOSmokeTrainer(
            tasks=[_task()],
            generator=MockGenerator(),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=PPOSmokeConfig(samples_per_task=4, max_steps=2, seed=20260601),
        ).run()

        self.assertEqual(len(result.metrics), 2)
        self.assertEqual(result.metrics[-1].rollout_count, 4)
        self.assertFalse(result.checkpoint_payload()["has_real_model_weights"])

    def test_summarize_ppo_step_computes_advantage_and_value_loss(self) -> None:
        result = PPOSmokeTrainer(
            tasks=[_task()],
            generator=MockGenerator(),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=PPOSmokeConfig(samples_per_task=4, max_steps=1, seed=20260603),
        ).run()
        metrics = summarize_ppo_step(1, list(result.last_rollouts))

        self.assertGreaterEqual(metrics.value_loss, 0.0)
        self.assertGreaterEqual(metrics.clipped_fraction, 0.0)
        self.assertLessEqual(metrics.clipped_fraction, 1.0)

    def test_algorithm_comparison_detects_matched_rollout_budget(self) -> None:
        comparison = compare_algorithm_metrics(
            {
                "rollout_count": 4,
                "mean_reward": 0.7,
                "execution_pass_rate": 0.5,
                "parse_failure_rate": 0.0,
                "informative_prompt_fraction": 1.0,
            },
            {
                "rollout_count": 4,
                "mean_reward": 0.6,
                "execution_pass_rate": 0.25,
                "parse_failure_rate": 0.0,
                "clipped_fraction": 0.1,
            },
        )

        self.assertTrue(comparison.rollout_budget_matched)
        self.assertAlmostEqual(comparison.mean_reward_delta_grpo_minus_ppo, 0.1)
        self.assertIn("PPO versus GRPO", comparison.to_markdown())

    def test_ppo_and_comparison_scripts_write_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            grpo_dir = Path(tmpdir) / "grpo"
            ppo_dir = Path(tmpdir) / "ppo"
            report_path = Path(tmpdir) / "compare.md"
            TaskRegistry([_task()]).to_jsonl(task_path)

            grpo_completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train_grpo_smoke.py"),
                    "--tasks",
                    str(task_path),
                    "--output-dir",
                    str(grpo_dir),
                    "--backend",
                    "mock",
                    "--group-size",
                    "4",
                    "--max-steps",
                    "1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            ppo_completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train_ppo_smoke.py"),
                    "--tasks",
                    str(task_path),
                    "--output-dir",
                    str(ppo_dir),
                    "--backend",
                    "mock",
                    "--samples-per-task",
                    "4",
                    "--max-steps",
                    "1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            compare_completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_training_smoke.py"),
                    "--grpo-metrics",
                    str(grpo_dir / "metrics.jsonl"),
                    "--ppo-metrics",
                    str(ppo_dir / "metrics.jsonl"),
                    "--output",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            ppo_checkpoint = json.loads(
                (ppo_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            report = report_path.read_text(encoding="utf-8")

        self.assertEqual(grpo_completed.returncode, 0, grpo_completed.stderr)
        self.assertEqual(ppo_completed.returncode, 0, ppo_completed.stderr)
        self.assertEqual(compare_completed.returncode, 0, compare_completed.stderr)
        self.assertEqual(ppo_checkpoint["kind"], "ppo_smoke_checkpoint")
        self.assertIn("PPO versus GRPO", report)
        self.assertIn("rollout_budget_matched", compare_completed.stdout)


if __name__ == "__main__":
    unittest.main()
