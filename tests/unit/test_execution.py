from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import ResourceLimits, Split, TaskSpec, TestSpec
from codeself.execution import DockerSandboxRunner, ExecutionJob, PhaseStatus, SandboxedTestRunner


def _task(
    *,
    public_code: str = "assert add_one(1) == 2",
    hidden_code: str = "assert add_one(0) == 1",
    timeout_seconds: float = 1.0,
) -> TaskSpec:
    return TaskSpec(
        task_id="execution/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.DEV,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code=public_code),),
        hidden_tests=(TestSpec(name="hidden", code=hidden_code),),
        resource_limits=ResourceLimits(timeout_seconds=timeout_seconds, memory_mb=256),
    )


class SandboxedTestRunnerTests(unittest.TestCase):
    def test_passing_solution_runs_public_and_hidden_tests(self) -> None:
        result = SandboxedTestRunner().run(_task(), "def add_one(x):\n    return x + 1")

        self.assertTrue(result.passed)
        self.assertEqual(result.status, PhaseStatus.PASSED)
        self.assertEqual(result.phase("public_tests").status, PhaseStatus.PASSED)
        self.assertEqual(result.phase("hidden_tests").status, PhaseStatus.PASSED)

    def test_public_failure_skips_hidden_by_default(self) -> None:
        result = SandboxedTestRunner().run(_task(), "def add_one(x):\n    return x")

        self.assertFalse(result.passed)
        self.assertEqual(result.phase("public_tests").status, PhaseStatus.FAILED)
        self.assertEqual(result.phase("hidden_tests").status, PhaseStatus.SKIPPED)

    def test_syntax_error_stops_before_execution(self) -> None:
        result = SandboxedTestRunner().run(_task(), "def add_one(x)\n    return x + 1")

        self.assertFalse(result.passed)
        self.assertEqual(result.status, PhaseStatus.PARSE_ERROR)
        self.assertIsNone(result.phase("import"))

    def test_security_scan_rejects_blocked_import(self) -> None:
        result = SandboxedTestRunner().run(_task(), "import os\n\ndef add_one(x):\n    return x + 1")

        self.assertFalse(result.passed)
        self.assertEqual(result.status, PhaseStatus.SECURITY_REJECTED)
        self.assertGreaterEqual(len(result.security_findings), 1)

    def test_timeout_is_reported(self) -> None:
        task = _task(
            public_code="assert spin() == 1",
            hidden_code="assert True",
            timeout_seconds=0.2,
        )
        result = SandboxedTestRunner().run(task, "def spin():\n    while True:\n        pass")

        self.assertFalse(result.passed)
        self.assertEqual(result.phase("public_tests").status, PhaseStatus.TIMEOUT)

    def test_test_phases_report_per_test_outcomes(self) -> None:
        task = TaskSpec(
            task_id="execution/multiple-tests",
            source="unit-test",
            prompt="Write add_one(x).",
            split=Split.DEV,
            entry_point="add_one",
            public_tests=(
                TestSpec(name="public-pass", code="assert add_one(1) == 2"),
                TestSpec(name="public-fail", code="assert add_one(2) == 4"),
            ),
            hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
            resource_limits=ResourceLimits(timeout_seconds=1.0, memory_mb=256),
        )
        result = SandboxedTestRunner().run(task, "def add_one(x):\n    return x + 1")
        public_phase = result.phase("public_tests")
        hidden_phase = result.phase("hidden_tests")

        self.assertFalse(result.passed)
        self.assertEqual(public_phase.status, PhaseStatus.FAILED)
        self.assertEqual(public_phase.tests_run, 2)
        self.assertEqual(
            [outcome.status for outcome in public_phase.test_outcomes],
            [PhaseStatus.PASSED, PhaseStatus.FAILED],
        )
        self.assertEqual(hidden_phase.status, PhaseStatus.SKIPPED)
        self.assertEqual(hidden_phase.test_outcomes[0].status, PhaseStatus.SKIPPED)

    def test_run_many_returns_results_in_input_order(self) -> None:
        first = _task(public_code="assert add_one(1) == 2")
        second = _task(public_code="assert add_one(1) == 3")
        runner = SandboxedTestRunner()

        results = runner.run_many(
            (
                ExecutionJob(first, "def add_one(x):\n    return x + 1", metadata={"name": "first"}),
                ExecutionJob(second, "def add_one(x):\n    return x + 1", metadata={"name": "second"}),
            ),
            max_workers=2,
        )

        self.assertEqual([item.metadata["name"] for item in results], ["first", "second"])
        self.assertTrue(results[0].result.passed)
        self.assertFalse(results[1].result.passed)


class DockerSandboxRunnerTests(unittest.TestCase):
    def test_docker_command_includes_final_eval_isolation_flags(self) -> None:
        command = DockerSandboxRunner().build_command(
            Path("/tmp/codeself-task"),
            ResourceLimits(timeout_seconds=1.0, memory_mb=128),
        )

        command_text = " ".join(command)
        self.assertIn("--network none", command_text)
        self.assertIn("--read-only", command)
        self.assertIn("--memory 128m", command_text)
        self.assertIn("--pids-limit", command)


if __name__ == "__main__":
    unittest.main()
