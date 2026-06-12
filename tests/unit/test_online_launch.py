from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402
from codeself.training import run_online_training_from_config  # noqa: E402


class OnlineTrainingLaunchTests(unittest.TestCase):
    def test_runs_grpo_online_training_from_single_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            output_dir = root / "artifacts"
            TaskRegistry([_task()]).to_jsonl(tasks_path)

            launch = run_online_training_from_config(
                _base_config(
                    "grpo",
                    tasks_path=tasks_path,
                    output_dir=output_dir,
                    training={
                        "max_batches": 1,
                        "optimizer": {"learning_rate": 0.05},
                        "loss": {"kl_beta": 0.0},
                    },
                )
            )
            summary = json.loads(
                (output_dir / "launch_summary.json").read_text(encoding="utf-8")
            )
            rollouts_exists = (output_dir / "cycle_0001" / "rollouts.jsonl").exists()
            checkpoint_exists = (output_dir / "cycle_0001" / "checkpoint.json").exists()

        self.assertEqual(launch.algorithm, "grpo")
        self.assertEqual(launch.model_backend, "toy")
        self.assertEqual(launch.task_count, 1)
        self.assertEqual(launch.cycle_count, 1)
        self.assertEqual(launch.total_rollouts, 2)
        self.assertEqual(launch.records_used, 2)
        self.assertEqual(summary["algorithm"], "grpo")
        self.assertEqual(summary["total_rollouts"], 2)
        self.assertTrue(rollouts_exists)
        self.assertTrue(checkpoint_exists)

    def test_run_online_training_script_runs_ppo_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            output_dir = root / "artifacts"
            config_path = root / "ppo_online.json"
            TaskRegistry([_task()]).to_jsonl(tasks_path)
            config = _base_config(
                "ppo",
                tasks_path=tasks_path,
                output_dir=output_dir,
                training={
                    "max_batches": 1,
                    "optimizer": {"learning_rate": 0.05},
                    "loss": {
                        "kl_beta": 0.0,
                        "normalize_advantages": False,
                        "value_clip_epsilon": None,
                    },
                },
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_online_training.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            summary = json.loads(
                (output_dir / "launch_summary.json").read_text(encoding="utf-8")
            )
            metrics_exists = (output_dir / "cycle_0001" / "metrics.jsonl").exists()
            checkpoint_exists = (output_dir / "cycle_0001" / "checkpoint.json").exists()

        self.assertIn("algorithm: ppo", result.stdout)
        self.assertIn("model_backend: toy", result.stdout)
        self.assertEqual(summary["algorithm"], "ppo")
        self.assertEqual(summary["total_rollouts"], 2)
        self.assertTrue(metrics_exists)
        self.assertTrue(checkpoint_exists)

    def test_rejects_ppo_transformers_without_value_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tasks_path = root / "tasks.jsonl"
            TaskRegistry([_task()]).to_jsonl(tasks_path)
            config = _base_config(
                "ppo",
                tasks_path=tasks_path,
                output_dir=root / "artifacts",
                model={"backend": "transformers", "name": "local-model"},
            )

            with self.assertRaisesRegex(ValueError, "value-head"):
                run_online_training_from_config(config)


def _base_config(
    algorithm: str,
    *,
    tasks_path: Path,
    output_dir: Path,
    training: dict[str, object] | None = None,
    model: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "data": {
            "tasks": str(tasks_path),
            "split": "train",
            "limit": 1,
            "skip_quality_checks": True,
        },
        "output": {"dir": str(output_dir)},
        "model": model or {"backend": "toy", "vocab_size": 512},
        "prompt": {"template": "direct_solution_v1"},
        "online": {"cycles": 1},
        "rollout": {
            "mode": "direct",
            "group_size": 2,
            "samples_per_task": 2,
            "max_prompt_tokens": 64,
            "max_response_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "batch": {"include_failed_parses": True},
        "training": training or {"max_batches": 1},
    }


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="agent/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(),
    )


if __name__ == "__main__":
    unittest.main()
