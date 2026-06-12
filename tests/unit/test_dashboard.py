from __future__ import annotations

import json
import subprocess
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
    compare_rollouts,
    evaluate_rollouts,
    render_evaluation_dashboard,
    write_comparison_report,
    write_evaluation_report,
)


class EvaluationDashboardTests(unittest.TestCase):
    def test_render_dashboard_includes_evaluation_and_comparison_sections(self) -> None:
        evaluation = evaluate_rollouts(
            _passing_rollouts("dashboard/pass") + _failing_rollouts("dashboard/fail"),
            ks=(1, 2),
            group_by=("metadata.difficulty",),
        )
        comparison = compare_rollouts(
            _failing_rollouts("dashboard/compare"),
            _passing_rollouts("dashboard/compare"),
            bootstrap_samples=50,
            permutation_samples=50,
            seed=1,
        )

        html = render_evaluation_dashboard(
            title="CodeSelf <Dashboard>",
            evaluations={"eval": evaluation.to_dict()},
            comparisons={"candidate": comparison.to_dict()},
        )

        self.assertIn("CodeSelf &lt;Dashboard&gt;", html)
        self.assertIn("Evaluation Runs", html)
        self.assertIn("Group Metrics", html)
        self.assertIn("Task Browser", html)
        self.assertIn("Paired Comparisons", html)
        self.assertIn("Task Flips", html)
        self.assertIn("<svg", html)
        with self.assertRaises(ValueError):
            render_evaluation_dashboard(title="empty", evaluations={})

    def test_render_dashboard_script_writes_html(self) -> None:
        records = (
            _passing_rollouts("dashboard/script-pass")
            + _failing_rollouts("dashboard/script-fail")
        )
        evaluation = evaluate_rollouts(
            records,
            ks=(1,),
            group_by=("metadata.difficulty",),
        )
        comparison = compare_rollouts(
            _failing_rollouts("dashboard/script-compare"),
            _passing_rollouts("dashboard/script-compare"),
            bootstrap_samples=50,
            permutation_samples=50,
            seed=1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            evaluation_path = tmp_path / "evaluation.json"
            comparison_path = tmp_path / "comparison.json"
            dashboard_path = tmp_path / "dashboard.html"
            artifact_dir = tmp_path / "online"
            _write_cycle(artifact_dir / "cycle_0001", _passing_rollouts("dashboard/curve"))
            write_evaluation_report(evaluation, evaluation_path)
            write_comparison_report(comparison, comparison_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "render_evaluation_dashboard.py"),
                    "--evaluation",
                    f"eval={evaluation_path}",
                    "--comparison",
                    f"cmp={comparison_path}",
                    "--curve",
                    f"online={artifact_dir}",
                    "--output",
                    str(dashboard_path),
                    "--title",
                    "Unit Dashboard",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            html = dashboard_path.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("wrote evaluation dashboard", completed.stdout)
        self.assertIn("evaluation_reports: 1", completed.stdout)
        self.assertIn("comparison_reports: 1", completed.stdout)
        self.assertIn("curve_reports: 1", completed.stdout)
        self.assertIn("Unit Dashboard", html)
        self.assertIn("Learning Curves", html)
        self.assertIn("cmp", html)


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
    records = generate_rollouts(
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
    for record in records:
        record.metadata["difficulty"] = "easy"
    return records


def _failing_rollouts(task_id: str):
    records = generate_rollouts(
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
    for record in records:
        record.metadata["difficulty"] = "hard"
    return records


def _write_cycle(cycle_dir: Path, rollouts) -> None:
    cycle_dir.mkdir(parents=True, exist_ok=True)

    write_rollouts_jsonl(rollouts, cycle_dir / "rollouts.jsonl")
    metric = {
        "phase": "ppo_model_training",
        "step": 1,
        "metrics": {
            "optimizer_step": True,
            "metrics": {
                "loss": 0.4,
                "policy_loss": -0.1,
                "value_loss": 0.5,
                "entropy_loss": -0.01,
                "kl_loss": 0.02,
                "mean_approx_kl": 0.03,
            },
        },
    }
    (cycle_dir / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    (cycle_dir / "checkpoint.json").write_text(
        json.dumps({"loop": {"optimizer_step_count": 1}}),
        encoding="utf-8",
    )
    write_evaluation_report(evaluate_rollouts(rollouts, ks=(1,)), cycle_dir / "evaluation.json")


if __name__ == "__main__":
    unittest.main()
