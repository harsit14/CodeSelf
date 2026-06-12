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
    AgentStep,
    AgentTrace,
    CodeGenerator,
    DIRECT_SOLUTION_TEMPLATE,
    GenerationRequest,
    GenerationResult,
    MockGenerator,
    StaticGenerator,
    extract_code,
    generate_self_debug_rollouts,
    generate_rollouts,
    read_rollouts_jsonl,
    rollout_record_from_trace,
    write_rollouts_jsonl,
)
from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="agent/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


class AgentRolloutTests(unittest.TestCase):
    def test_prompt_template_does_not_include_hidden_tests(self) -> None:
        prompt = DIRECT_SOLUTION_TEMPLATE.render(_task())

        self.assertIn("Write add_one", prompt)
        self.assertNotIn("assert add_one(0) == 1", prompt)

    def test_parser_prefers_python_fenced_block(self) -> None:
        parsed = extract_code(
            "Here is code:\n"
            "```text\nnot python\n```\n"
            "```python\ndef add_one(x):\n    return x + 1\n```"
        )

        self.assertTrue(parsed.ok)
        self.assertIn("def add_one", parsed.code)
        self.assertEqual(parsed.parser, "fenced:python:2")

    def test_parser_reports_syntax_error(self) -> None:
        parsed = extract_code("```python\ndef bad(:\n    pass\n```")

        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.status.value, "syntax_error")

    def test_generate_rollouts_with_static_generator_scores_solution(self) -> None:
        records = generate_rollouts(
            [_task()],
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x + 1\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=1,
            seed=1,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            include_hidden=True,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].reward["reward"], 1.0)
        self.assertTrue(records[0].execution["passed"])

    def test_generate_rollouts_accepts_custom_reward_scorer(self) -> None:
        from codeself.rewards import ConfigurableRewardScorer, RewardModeConfig

        task = TaskSpec(
            task_id="agent/fractional",
            source="unit-test",
            prompt="Write add_one(x).",
            split=Split.TRAIN,
            entry_point="add_one",
            public_tests=(
                TestSpec(name="public-pass", code="assert add_one(1) == 2"),
                TestSpec(name="public-fail", code="assert add_one(2) == 4"),
            ),
            hidden_tests=(),
        )
        records = generate_rollouts(
            [task],
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x + 1\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=1,
            seed=1,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            include_hidden=True,
            scorer=ConfigurableRewardScorer(
                RewardModeConfig(mode="fractional_pass_rate", name="fractional")
            ),
            metadata={"reward_mode": "fractional_pass_rate"},
        )

        self.assertEqual(records[0].reward["reward"], 0.5)
        self.assertEqual(records[0].metadata["reward_mode"], "fractional_pass_rate")

    def test_generate_self_debug_rollouts_returns_training_records(self) -> None:
        result = generate_self_debug_rollouts(
            [_task()],
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=1,
            seed=3,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            include_hidden=True,
            max_revisions=1,
            revision_reward_discount=0.5,
            revision_reward_step_penalty=0.1,
            metadata={"ablation": "self_debug"},
        )

        record = result.records[0]

        self.assertEqual(len(result.traces), 1)
        self.assertAlmostEqual(record.reward["reward"], 0.4)
        self.assertEqual(
            record.reward["metrics"]["self_debug_reward_discount_mode"],
            "positive_only",
        )
        self.assertAlmostEqual(
            record.reward["metrics"]["self_debug_step_penalty_total"],
            0.1,
        )
        self.assertTrue(record.execution["passed"])
        self.assertIn("return x + 1", record.raw_completion)
        self.assertEqual(record.metadata["rollout_mode"], "self_debug")
        self.assertEqual(record.metadata["revision_count"], 1)
        self.assertEqual(record.metadata["revision_reward_discount_mode"], "positive_only")
        self.assertAlmostEqual(record.metadata["revision_reward_step_penalty"], 0.1)
        self.assertEqual(record.metadata["ablation"], "self_debug")
        self.assertEqual(result.analysis.final_pass_rate, 1.0)

    def test_self_debug_reward_shaping_can_discount_all_rewards(self) -> None:
        trace = AgentTrace(
            task_id="agent/failing",
            prompt_template="direct_solution_v1",
            prompt="Write add_one(x).",
            raw_completion="def add_one(x):\n    return x",
            initial_code="def add_one(x):\n    return x",
            final_code="def add_one(x):\n    return x",
            steps=(
                AgentStep(
                    kind="generation",
                    parsed_status="ok",
                    metadata={"backend": "static", "model_name": "unit"},
                ),
                AgentStep(kind="revision", parsed_status="ok"),
            ),
            final_result={"passed": False},
            final_reward={"reward": -1.0, "metrics": {"original": -1.0}},
            metadata={
                "seed": 9,
                "include_hidden": True,
                "max_revisions": 1,
                "use_rule_based_repair": True,
            },
        )

        record = rollout_record_from_trace(
            trace,
            revision_reward_discount=0.5,
            revision_reward_discount_mode="all",
            revision_reward_step_penalty=0.1,
        )

        self.assertAlmostEqual(record.reward["reward"], -0.6)
        self.assertAlmostEqual(
            record.reward["metrics"]["self_debug_discounted_reward_before_penalty"],
            -0.5,
        )
        self.assertEqual(record.reward["metrics"]["self_debug_reward_discount_mode"], "all")
        self.assertAlmostEqual(
            record.reward["metrics"]["self_debug_step_penalty_total"],
            0.1,
        )

    def test_generate_self_debug_rollouts_can_use_model_revision(self) -> None:
        generator = _TwoStageRevisionGenerator()
        result = generate_self_debug_rollouts(
            [_task()],
            generator=generator,
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=1,
            seed=3,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            revision_max_new_tokens=32,
            revision_temperature=0.3,
            revision_top_p=0.8,
            include_hidden=True,
            max_revisions=1,
            use_rule_based_repair=False,
            revision_strategy="model",
        )

        record = result.records[0]

        self.assertTrue(record.execution["passed"])
        self.assertEqual(record.metadata["revision_strategy"], "model")
        self.assertEqual(record.metadata["revision_count"], 1)
        self.assertEqual(record.metadata["revision_max_new_tokens"], 32)
        self.assertEqual(record.metadata["revision_temperature"], 0.3)
        self.assertEqual(record.metadata["revision_top_p"], 0.8)
        self.assertEqual(record.generation_metadata["stage"], "initial")
        self.assertEqual(result.config.resolved_revision_max_new_tokens, 32)
        self.assertEqual(generator.requests[1].max_new_tokens, 32)

    def test_rollout_jsonl_round_trip(self) -> None:
        records = generate_rollouts(
            [_task()],
            generator=MockGenerator(),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=2,
            seed=1,
            max_new_tokens=128,
            temperature=0.8,
            top_p=0.95,
            include_hidden=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            write_rollouts_jsonl(records, path)
            restored = read_rollouts_jsonl(path)

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].task_id, "agent/add-one")

    def test_run_rollouts_script_writes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            output_path = Path(tmpdir) / "rollouts.jsonl"
            TaskRegistry([_task()]).to_jsonl(task_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_rollouts.py"),
                    "--tasks",
                    str(task_path),
                    "--output",
                    str(output_path),
                    "--backend",
                    "mock",
                    "--samples-per-task",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.exists())
            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(len(records), 2)
        self.assertIn("mean_reward", completed.stdout)

    def test_run_rollouts_script_writes_self_debug_records_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            completion_path = Path(tmpdir) / "completion.txt"
            output_path = Path(tmpdir) / "rollouts.jsonl"
            trace_path = Path(tmpdir) / "traces.jsonl"
            TaskRegistry([_task()]).to_jsonl(task_path)
            completion_path.write_text(
                "```python\ndef add_one(x):\n    return x\n```",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_rollouts.py"),
                    "--tasks",
                    str(task_path),
                    "--output",
                    str(output_path),
                    "--rollout-mode",
                    "self_debug",
                    "--trace-output",
                    str(trace_path),
                    "--backend",
                    "static",
                    "--static-completion-file",
                    str(completion_path),
                    "--samples-per-task",
                    "1",
                    "--max-revisions",
                    "1",
                    "--revision-max-new-tokens",
                    "33",
                    "--revision-temperature",
                    "0.25",
                    "--revision-top-p",
                    "0.8",
                    "--revision-reward-discount",
                    "0.5",
                    "--revision-reward-discount-mode",
                    "all",
                    "--revision-reward-step-penalty",
                    "0.1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            traces = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(records[0]["metadata"]["rollout_mode"], "self_debug")
        self.assertEqual(records[0]["metadata"]["revision_count"], 1)
        self.assertEqual(records[0]["metadata"]["revision_max_new_tokens"], 33)
        self.assertEqual(records[0]["metadata"]["revision_temperature"], 0.25)
        self.assertEqual(records[0]["metadata"]["revision_top_p"], 0.8)
        self.assertEqual(records[0]["metadata"]["revision_reward_discount_mode"], "all")
        self.assertEqual(records[0]["metadata"]["revision_reward_step_penalty"], 0.1)
        self.assertEqual(records[0]["reward"]["reward"], 0.4)
        self.assertEqual(
            records[0]["reward"]["metrics"]["self_debug_reward_discount_mode"],
            "all",
        )
        self.assertEqual(
            records[0]["reward"]["metrics"]["self_debug_step_penalty_total"],
            0.1,
        )
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["metadata"]["revision_max_new_tokens"], 33)
        self.assertIn("rollout_mode: self_debug", completed.stdout)
        self.assertIn("self_debug_revision_rate: 1.0000", completed.stdout)

    def test_run_rollouts_script_reads_config_and_reward_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task_path = tmp_path / "tasks.jsonl"
            output_path = tmp_path / "rollouts.jsonl"
            completion_path = tmp_path / "completion.txt"
            config_path = tmp_path / "rollout.json"
            task = TaskSpec(
                task_id="agent/config-fractional",
                source="unit-test",
                prompt="Write add_one(x).",
                split=Split.TRAIN,
                entry_point="add_one",
                public_tests=(
                    TestSpec(name="public-pass", code="assert add_one(1) == 2"),
                    TestSpec(name="public-fail", code="assert add_one(2) == 4"),
                ),
                hidden_tests=(),
            )
            TaskRegistry([task]).to_jsonl(task_path)
            completion_path.write_text(
                "```python\ndef add_one(x):\n    return x + 1\n```",
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps(
                    {
                        "experiment": {"seed": 7},
                        "data": {"tasks": str(task_path), "split": "train"},
                        "prompt": {"template": "direct_solution_v1"},
                        "generation": {
                            "backend": "static",
                            "static_completion_file": str(completion_path),
                            "samples_per_task": 1,
                        },
                        "execution": {"include_hidden": True},
                        "reward": {"mode": "fractional_pass_rate"},
                        "output": {"rollouts": str(output_path)},
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_rollouts.py"),
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(records[0]["reward"]["reward_name"], "reward_fractional_pass_rate")
        self.assertEqual(records[0]["reward"]["reward"], 0.5)
        self.assertEqual(records[0]["metadata"]["reward_mode"], "fractional_pass_rate")

    def test_run_rollouts_script_fails_on_dataset_quality_issue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            task_path = tmp_path / "tasks.jsonl"
            output_path = tmp_path / "rollouts.jsonl"
            TaskRegistry(
                [
                    TaskSpec(
                        task_id="agent/train-duplicate",
                        source="unit-test",
                        prompt="Return x plus one.",
                        split=Split.TRAIN,
                        entry_point="add_one",
                        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
                    ),
                    TaskSpec(
                        task_id="agent/dev-duplicate",
                        source="unit-test",
                        prompt="Return x plus one.",
                        split=Split.DEV,
                        entry_point="add_one",
                        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
                    ),
                ]
            ).to_jsonl(task_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_rollouts.py"),
                    "--tasks",
                    str(task_path),
                    "--output",
                    str(output_path),
                    "--backend",
                    "mock",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("dataset quality checks failed", completed.stdout)
        self.assertIn("contamination", completed.stdout)


class _TwoStageRevisionGenerator(CodeGenerator):
    backend_name = "unit"
    model_name = "two-stage-revision"

    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests.append(request)
        if "Public-test feedback:" in request.prompt:
            completion = "```python\ndef add_one(x):\n    return x + 1\n```"
            stage = "revision"
        else:
            completion = "```python\ndef add_one(x):\n    return x\n```"
            stage = "initial"
        return GenerationResult(
            text=completion,
            backend=self.backend_name,
            model_name=self.model_name,
            metadata={"stage": stage},
        )


if __name__ == "__main__":
    unittest.main()
