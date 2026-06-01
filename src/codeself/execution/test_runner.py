"""Task-level test runner built on top of the subprocess sandbox."""

from __future__ import annotations

import ast
import time

from codeself.datasets import TaskSpec, TestSpec
from codeself.execution.results import PhaseResult, PhaseStatus, TaskRunResult
from codeself.execution.security_scan import scan_python_code
from codeself.execution.subprocess_runner import SubprocessSandboxRunner


class SandboxedTestRunner:
    """Runs generated Python code through syntax, import, public, and hidden phases."""

    def __init__(
        self,
        sandbox: SubprocessSandboxRunner | None = None,
        *,
        run_hidden_after_public_failure: bool = False,
    ) -> None:
        self.sandbox = sandbox or SubprocessSandboxRunner()
        self.run_hidden_after_public_failure = run_hidden_after_public_failure

    def run(self, task: TaskSpec, solution_code: str, *, include_hidden: bool = True) -> TaskRunResult:
        candidate_code = _candidate_code(task.starter_code, solution_code)
        phases: list[PhaseResult] = []

        syntax_phase = _syntax_phase(candidate_code)
        phases.append(syntax_phase)
        if not syntax_phase.passed:
            return TaskRunResult(task_id=task.task_id, status=syntax_phase.status, phases=tuple(phases))

        scan = scan_python_code(candidate_code)
        if not scan.allowed:
            phase = PhaseResult(
                name="security",
                status=PhaseStatus.SECURITY_REJECTED,
                error="; ".join(finding.message for finding in scan.findings),
            )
            phases.append(phase)
            return TaskRunResult(
                task_id=task.task_id,
                status=PhaseStatus.SECURITY_REJECTED,
                phases=tuple(phases),
                security_findings=scan.findings,
            )

        import_phase = self.sandbox.run_phase(
            phase_name="import",
            candidate_code=candidate_code,
            phase_code="pass",
            limits=task.resource_limits,
            tests_run=0,
        )
        phases.append(import_phase)
        if not import_phase.passed:
            return TaskRunResult(task_id=task.task_id, status=import_phase.status, phases=tuple(phases))

        public_phase = self.sandbox.run_phase(
            phase_name="public_tests",
            candidate_code=candidate_code,
            phase_code=_tests_code(task.public_tests),
            limits=task.resource_limits,
            tests_run=len(task.public_tests),
        )
        phases.append(public_phase)

        if include_hidden:
            if public_phase.passed or self.run_hidden_after_public_failure:
                hidden_phase = self.sandbox.run_phase(
                    phase_name="hidden_tests",
                    candidate_code=candidate_code,
                    phase_code=_tests_code(task.hidden_tests),
                    limits=task.resource_limits,
                    tests_run=len(task.hidden_tests),
                )
            else:
                hidden_phase = PhaseResult(
                    name="hidden_tests",
                    status=PhaseStatus.SKIPPED,
                    error="skipped because public tests failed",
                    tests_run=len(task.hidden_tests),
                )
            phases.append(hidden_phase)

        final_status = _final_status(phases, include_hidden)
        return TaskRunResult(task_id=task.task_id, status=final_status, phases=tuple(phases))


def _candidate_code(starter_code: str, solution_code: str) -> str:
    return f"{starter_code.rstrip()}\n\n{solution_code.strip()}\n"


def _syntax_phase(code: str) -> PhaseResult:
    start = time.monotonic()
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return PhaseResult(
            name="syntax",
            status=PhaseStatus.PARSE_ERROR,
            duration_seconds=time.monotonic() - start,
            error=f"{exc.msg} at line {exc.lineno}",
        )
    return PhaseResult(
        name="syntax",
        status=PhaseStatus.PASSED,
        duration_seconds=time.monotonic() - start,
    )


def _tests_code(tests: tuple[TestSpec, ...]) -> str:
    if not tests:
        return "pass"
    return "\n\n".join(test.code for test in tests)


def _final_status(phases: list[PhaseResult], include_hidden: bool) -> PhaseStatus:
    phase_by_name = {phase.name: phase for phase in phases}
    public_phase = phase_by_name.get("public_tests")
    hidden_phase = phase_by_name.get("hidden_tests")
    if public_phase is None or not public_phase.passed:
        return public_phase.status if public_phase is not None else PhaseStatus.RUNTIME_ERROR
    if include_hidden and hidden_phase is not None and not hidden_phase.passed:
        return hidden_phase.status
    return PhaseStatus.PASSED
