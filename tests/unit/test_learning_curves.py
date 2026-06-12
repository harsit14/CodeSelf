from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    DIRECT_SOLUTION_TEMPLATE,
    StaticGenerator,
    generate_rollouts,
    write_rollouts_jsonl,
)
from codeself.datasets import Split, TaskSpec, TestSpec  # noqa: E402
from codeself.evaluation import (  # noqa: E402
    collect_learning_curve,
    collect_learning_curves,
    evaluate_rollouts,
    write_evaluation_report,
)


class LearningCurveTests(unittest.TestCase):
    def test_collect_learning_curve_reads_online_cycle_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "online"
            _write_cycle(
                artifact_dir / "cycle_0001",
                rollouts=_passing_rollouts("curve/a"),
                loss=0.5,
                policy_loss=-0.2,
                value_loss=0.7,
                kl_loss=0.01,
                approx_kl=0.02,
            )
            _write_cycle(
                artifact_dir / "cycle_0002",
                rollouts=_failing_rollouts("curve/b"),
                loss=0.8,
                policy_loss=0.1,
                value_loss=0.6,
                kl_loss=0.03,
                approx_kl=0.04,
            )

            run = collect_learning_curve("run", artifact_dir)
            payloads = collect_learning_curves({"run": artifact_dir})

        self.assertEqual(run.label, "run")
        self.assertEqual(run.cycle_count, 2)
        self.assertEqual(run.total_rollouts, 4)
        self.assertEqual(run.total_optimizer_steps, 2)
        self.assertEqual(run.points[0].cycle, 1)
        self.assertAlmostEqual(run.points[0].train_pass_at_1, 1.0)
        self.assertAlmostEqual(run.points[1].train_pass_at_1, 0.0)
        self.assertAlmostEqual(run.points[0].train_loss, 0.5)
        self.assertAlmostEqual(run.points[1].mean_approx_kl, 0.04)
        self.assertAlmostEqual(run.points[0].eval_pass_at_1 or 0.0, 1.0)
        self.assertIn("points", payloads["run"])
        with self.assertRaises(ValueError):
            collect_learning_curve("missing", Path(tmpdir) / "missing")


def _write_cycle(
    cycle_dir: Path,
    *,
    rollouts,
    loss: float,
    policy_loss: float,
    value_loss: float,
    kl_loss: float,
    approx_kl: float,
) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)
    write_rollouts_jsonl(rollouts, cycle_dir / "rollouts.jsonl")
    metrics = {
        "phase": "ppo_model_training",
        "step": 1,
        "metrics": {
            "optimizer_step": True,
            "metrics": {
                "loss": loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": -0.01,
                "kl_loss": kl_loss,
                "mean_approx_kl": approx_kl,
            },
        },
    }
    (cycle_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    checkpoint = {"loop": {"optimizer_step_count": 1}}
    (cycle_dir / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    write_evaluation_report(evaluate_rollouts(rollouts, ks=(1,)), cycle_dir / "evaluation.json")


def _task(task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TEST_PUBLIC,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


def _passing_rollouts(task_id: str):
    return generate_rollouts(
        [_task(task_id)],
        generator=StaticGenerator("```python\ndef add_one(x):\n    return x + 1\n```"),
        prompt_template=DIRECT_SOLUTION_TEMPLATE,
        samples_per_task=2,
        seed=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        include_hidden=True,
    )


def _failing_rollouts(task_id: str):
    return generate_rollouts(
        [_task(task_id)],
        generator=StaticGenerator("```python\ndef add_one(x):\n    return x\n```"),
        prompt_template=DIRECT_SOLUTION_TEMPLATE,
        samples_per_task=2,
        seed=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        include_hidden=True,
    )


if __name__ == "__main__":
    unittest.main()
