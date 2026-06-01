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
from codeself.training import (  # noqa: E402
    ExperimentRunSummary,
    ExperimentVariant,
    aggregate_by_variant_name,
    build_grpo_experiment_matrix,
    select_best_run,
)


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="matrix/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


def _summary(
    variant: ExperimentVariant,
    *,
    mean_reward: float,
    informative: float,
    stopped: bool = False,
) -> ExperimentRunSummary:
    return ExperimentRunSummary(
        variant=variant,
        output_dir="/tmp/run",
        steps_completed=3,
        mean_reward=mean_reward,
        informative_prompt_fraction=informative,
        reward_variance_prompt_fraction=informative,
        execution_pass_rate=mean_reward,
        parse_failure_rate=0.0,
        stopped_early=stopped,
    )


class ExperimentMatrixTests(unittest.TestCase):
    def test_build_matrix_crosses_sweeps(self) -> None:
        variants = build_grpo_experiment_matrix(
            seeds=(1, 2),
            group_sizes=(4, 8),
            kl_betas=(0.02,),
            curriculum_modes=("static", "informative"),
            reward_names=("reward_v0_correctness",),
            max_steps=2,
        )

        self.assertEqual(len(variants), 8)
        self.assertEqual(len({variant.slug for variant in variants}), 8)

    def test_select_best_run_ignores_stopped_when_required(self) -> None:
        weak = ExperimentVariant(name="weak", seed=1, group_size=4, kl_beta=0.02)
        stopped = ExperimentVariant(name="stopped", seed=2, group_size=4, kl_beta=0.02)
        summaries = [
            _summary(weak, mean_reward=0.5, informative=0.5),
            _summary(stopped, mean_reward=1.0, informative=1.0, stopped=True),
        ]

        best = select_best_run(summaries)

        self.assertEqual(best.variant.name, "weak")

    def test_select_best_run_can_require_informative_prompts(self) -> None:
        high_reward = ExperimentVariant(name="high", seed=1, group_size=4, kl_beta=0.02)
        learnable = ExperimentVariant(name="learnable", seed=2, group_size=4, kl_beta=0.02)
        summaries = [
            _summary(high_reward, mean_reward=1.0, informative=0.0),
            _summary(learnable, mean_reward=0.7, informative=0.5),
        ]

        best = select_best_run(summaries, minimum_informative_prompt_fraction=0.1)

        self.assertEqual(best.variant.name, "learnable")

    def test_aggregate_by_variant_name_averages_repeated_seeds(self) -> None:
        variant_a = ExperimentVariant(name="same", seed=1, group_size=4, kl_beta=0.02)
        variant_b = ExperimentVariant(name="same", seed=2, group_size=4, kl_beta=0.02)

        aggregate = aggregate_by_variant_name(
            [
                _summary(variant_a, mean_reward=0.2, informative=0.4),
                _summary(variant_b, mean_reward=0.8, informative=0.6),
            ]
        )

        values = next(iter(aggregate.values()))
        self.assertEqual(values["runs"], 2)
        self.assertAlmostEqual(values["mean_reward"], 0.5)
        self.assertAlmostEqual(values["informative_prompt_fraction"], 0.5)

    def test_run_experiment_matrix_script_writes_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            output_dir = Path(tmpdir) / "matrix"
            TaskRegistry([_task()]).to_jsonl(task_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_experiment_matrix.py"),
                    "--tasks",
                    str(task_path),
                    "--output-dir",
                    str(output_dir),
                    "--backend",
                    "mock",
                    "--seeds",
                    "20260601,20260602",
                    "--group-sizes",
                    "4",
                    "--kl-betas",
                    "0.02",
                    "--curriculum-modes",
                    "static",
                    "--max-steps",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            selection = json.loads(
                (output_dir / "selected_checkpoint.json").read_text(encoding="utf-8")
            )
            table = (output_dir / "experiment_table.md").read_text(encoding="utf-8")

        self.assertEqual(selection["primary_metric"], "mean_reward")
        self.assertIn("GRPO Experiment Table", table)
        self.assertIn("ran 2 experiment variants", completed.stdout)


if __name__ == "__main__":
    unittest.main()
