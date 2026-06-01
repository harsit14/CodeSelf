from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import DIRECT_SOLUTION_TEMPLATE, MockGenerator, StaticGenerator  # noqa: E402
from codeself.agent.rollouts import generate_rollouts  # noqa: E402
from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402
from codeself.training import (  # noqa: E402
    GRPOSmokeConfig,
    GRPOSmokeTrainer,
    compute_group_advantages,
)


def _task(task_id: str = "grpo/add-one") -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


class GRPOSmokeTests(unittest.TestCase):
    def test_compute_group_advantages_normalizes_mixed_rewards(self) -> None:
        records = generate_rollouts(
            [_task()],
            generator=MockGenerator(),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=4,
            seed=20260603,
            max_new_tokens=128,
            temperature=0.8,
            top_p=0.95,
            include_hidden=True,
        )

        groups = compute_group_advantages(records)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].total_samples, 4)
        self.assertGreater(groups[0].reward_std, 0)
        self.assertAlmostEqual(sum(groups[0].advantages), 0.0, places=7)

    def test_smoke_trainer_runs_steps_and_reports_informative_fraction(self) -> None:
        trainer = GRPOSmokeTrainer(
            tasks=[_task()],
            generator=MockGenerator(),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=GRPOSmokeConfig(group_size=4, max_steps=2, seed=20260601),
        )

        result = trainer.run()

        self.assertEqual(len(result.metrics), 2)
        self.assertEqual(result.metrics[-1].rollout_count, 4)
        self.assertGreaterEqual(result.metrics[-1].informative_prompt_fraction, 0.0)
        self.assertFalse(result.checkpoint_payload()["has_real_model_weights"])

    def test_smoke_trainer_stops_when_informative_fraction_too_low(self) -> None:
        trainer = GRPOSmokeTrainer(
            tasks=[_task()],
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x + 1\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=GRPOSmokeConfig(
                group_size=3,
                max_steps=5,
                stop_if_informative_prompt_fraction_below=0.1,
            ),
        )

        result = trainer.run()

        self.assertEqual(len(result.metrics), 1)
        self.assertTrue(result.metrics[0].stopped_early)
        self.assertEqual(result.metrics[0].informative_prompt_fraction, 0.0)

    def test_train_grpo_smoke_script_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            output_dir = Path(tmpdir) / "smoke"
            TaskRegistry([_task()]).to_jsonl(task_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train_grpo_smoke.py"),
                    "--tasks",
                    str(task_path),
                    "--output-dir",
                    str(output_dir),
                    "--backend",
                    "mock",
                    "--group-size",
                    "4",
                    "--max-steps",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            checkpoint = json.loads(
                (output_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
            )
            metric_lines = (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(metric_lines), 2)
        self.assertEqual(checkpoint["kind"], "grpo_smoke_checkpoint")
        self.assertIn("informative_prompt_fraction", completed.stdout)


if __name__ == "__main__":
    unittest.main()
