from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import ResourceLimits, Split, TaskSpec, TestSpec
from codeself.execution import DockerSandboxRunner, PhaseStatus, SandboxedTestRunner


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
