from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    DIRECT_SOLUTION_TEMPLATE,
    AgentLoopConfig,
    CodeGenerator,
    GenerationRequest,
    GenerationResult,
    SELF_DEBUG_REVISION_TEMPLATE,
    SelfDebugAgentLoop,
    StaticGenerator,
    analyze_traces,
    infer_simple_revision,
    read_agent_traces_jsonl,
    run_agentic_tasks,
    write_agent_traces_jsonl,
)
from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="agentic/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


class AgenticLoopTests(unittest.TestCase):
    def test_rule_based_revision_infers_public_test_delta(self) -> None:
        revised = infer_simple_revision(_task(), "def add_one(x):\n    return x\n", "failed")

        self.assertIn("return x + 1", revised)

    def test_self_debug_loop_repairs_public_failure_and_passes_final(self) -> None:
        loop = SelfDebugAgentLoop(
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=AgentLoopConfig(max_revisions=1),
        )

        trace = loop.run_task(_task())

        self.assertTrue(trace.final_passed)
        self.assertEqual(trace.revision_count, 1)
        self.assertGreaterEqual(trace.tool_call_count, 3)
        self.assertIn("return x + 1", trace.final_code)

    def test_self_debug_loop_without_revision_fails_final(self) -> None:
        loop = SelfDebugAgentLoop(
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=AgentLoopConfig(max_revisions=0),
        )

        trace = loop.run_task(_task())

        self.assertFalse(trace.final_passed)
        self.assertEqual(trace.revision_count, 0)

    def test_self_debug_loop_can_use_model_revision_prompt(self) -> None:
        generator = _TwoStageRevisionGenerator()
        loop = SelfDebugAgentLoop(
            generator=generator,
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=AgentLoopConfig(
                max_revisions=1,
                use_rule_based_repair=False,
                use_model_revision=True,
            ),
        )

        trace = loop.run_task(_task())
        revision_step = next(step for step in trace.steps if step.kind == "revision")

        self.assertTrue(trace.final_passed)
        self.assertEqual(trace.revision_count, 1)
        self.assertEqual(revision_step.metadata["revision_source"], "model")
        self.assertEqual(
            revision_step.metadata["revision_prompt_template"],
            SELF_DEBUG_REVISION_TEMPLATE.name,
        )
        self.assertIn("Public-test feedback:", revision_step.metadata["revision_prompt"])
        self.assertIn("Current code:", revision_step.metadata["revision_prompt"])
        self.assertNotIn("assert add_one(0) == 1", revision_step.metadata["revision_prompt"])
        self.assertIn("return x + 1", trace.final_code)

    def test_model_revision_can_use_separate_sampling_settings(self) -> None:
        generator = _TwoStageRevisionGenerator()
        loop = SelfDebugAgentLoop(
            generator=generator,
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=AgentLoopConfig(
                max_revisions=1,
                max_new_tokens=64,
                temperature=0.9,
                top_p=0.95,
                revision_max_new_tokens=17,
                revision_temperature=0.2,
                revision_top_p=0.7,
                use_rule_based_repair=False,
                use_model_revision=True,
            ),
        )

        trace = loop.run_task(_task())
        revision_step = next(step for step in trace.steps if step.kind == "revision")

        self.assertEqual(generator.requests[0].max_new_tokens, 64)
        self.assertEqual(generator.requests[0].temperature, 0.9)
        self.assertEqual(generator.requests[1].max_new_tokens, 17)
        self.assertEqual(generator.requests[1].temperature, 0.2)
        self.assertEqual(generator.requests[1].top_p, 0.7)
        self.assertEqual(revision_step.metadata["revision_request_max_new_tokens"], 17)
        self.assertEqual(revision_step.metadata["revision_request_temperature"], 0.2)
        self.assertEqual(trace.metadata["revision_max_new_tokens"], 17)
        self.assertTrue(trace.final_passed)

    def test_trace_round_trip_and_analysis(self) -> None:
        loop = SelfDebugAgentLoop(
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            config=AgentLoopConfig(max_revisions=1),
        )
        traces = run_agentic_tasks([_task()], loop=loop)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traces.jsonl"
            write_agent_traces_jsonl(traces, path)
            restored = read_agent_traces_jsonl(path)

        analysis = analyze_traces(restored)
        self.assertEqual(len(restored), 1)
        self.assertEqual(analysis.trace_count, 1)
        self.assertEqual(analysis.final_pass_rate, 1.0)
        self.assertEqual(analysis.revision_rate, 1.0)

    def test_run_agentic_and_view_trace_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            completion_path = Path(tmpdir) / "completion.txt"
            trace_path = Path(tmpdir) / "traces.jsonl"
            report_path = Path(tmpdir) / "report.md"
            view_path = Path(tmpdir) / "trace.md"
            TaskRegistry([_task()]).to_jsonl(task_path)
            completion_path.write_text(
                "```python\ndef add_one(x):\n    return x\n```",
                encoding="utf-8",
            )

            run_completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_agentic.py"),
                    "--tasks",
                    str(task_path),
                    "--trace-output",
                    str(trace_path),
                    "--report-output",
                    str(report_path),
                    "--backend",
                    "static",
                    "--static-completion-file",
                    str(completion_path),
                    "--max-revisions",
                    "1",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            view_completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "view_agent_trace.py"),
                    "--traces",
                    str(trace_path),
                    "--output",
                    str(view_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            report = report_path.read_text(encoding="utf-8")
            view = view_path.read_text(encoding="utf-8")

        self.assertEqual(run_completed.returncode, 0, run_completed.stderr)
        self.assertEqual(view_completed.returncode, 0, view_completed.stderr)
        self.assertIn("final_pass_rate: 1.0000", run_completed.stdout)
        self.assertIn("Agentic Strategy Analysis", report)
        self.assertIn("Agent Trace", view)


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
