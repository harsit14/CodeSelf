"""Task-level test runner built on top of the subprocess sandbox."""

from __future__ import annotations

import ast
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from codeself.datasets import TaskSpec, TestSpec
from codeself.execution.results import PhaseResult, PhaseStatus, TaskRunResult, TestOutcome
from codeself.execution.security_scan import scan_python_code
from codeself.execution.subprocess_runner import SubprocessSandboxRunner


@dataclass(frozen=True)
class ExecutionJob:
    """One candidate execution request for batch sandbox runs."""

    task: TaskSpec
    solution_code: str
    include_hidden: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchExecutionResult:
    """Result for one item from a batch sandbox run."""

    job_index: int
    task_id: str
    result: TaskRunResult
    metadata: dict[str, Any] = field(default_factory=dict)


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

        public_phase = self._run_tests_phase(
            phase_name="public_tests",
            candidate_code=candidate_code,
            tests=task.public_tests,
            limits=task.resource_limits,
        )
        phases.append(public_phase)

        if include_hidden:
            if public_phase.passed or self.run_hidden_after_public_failure:
                hidden_phase = self._run_tests_phase(
                    phase_name="hidden_tests",
                    candidate_code=candidate_code,
                    tests=task.hidden_tests,
                    limits=task.resource_limits,
                )
            else:
                hidden_phase = _skipped_tests_phase(task.hidden_tests)
            phases.append(hidden_phase)

        final_status = _final_status(phases, include_hidden)
        return TaskRunResult(task_id=task.task_id, status=final_status, phases=tuple(phases))

    def run_many(
        self,
        jobs: Iterable[ExecutionJob],
        *,
        max_workers: int = 4,
    ) -> list[BatchExecutionResult]:
        """Run candidate programs concurrently with a worker pool.

        Results are returned in the same order as the input jobs.
        """

        job_list = list(jobs)
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not job_list:
            return []

        results: dict[int, BatchExecutionResult] = {}
        worker_count = min(max_workers, len(job_list))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {
                executor.submit(
                    self.run,
                    job.task,
                    job.solution_code,
                    include_hidden=job.include_hidden,
                ): index
                for index, job in enumerate(job_list)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                job = job_list[index]
                result = future.result()
                results[index] = BatchExecutionResult(
                    job_index=index,
                    task_id=job.task.task_id,
                    result=result,
                    metadata=job.metadata,
                )

        return [results[index] for index in range(len(job_list))]

    def _run_tests_phase(
        self,
        *,
        phase_name: str,
        candidate_code: str,
        tests: tuple[TestSpec, ...],
        limits,
    ) -> PhaseResult:
        if not tests:
            return PhaseResult(name=phase_name, status=PhaseStatus.PASSED, tests_run=0)

        phase_results = [
            self.sandbox.run_phase(
                phase_name=f"{phase_name}:{test.name}",
                candidate_code=candidate_code,
                phase_code=test.code,
                limits=limits,
                tests_run=1,
            )
            for test in tests
        ]
        outcomes = tuple(
            TestOutcome(
                name=test.name,
                status=result.status,
                duration_seconds=result.duration_seconds,
                stdout=result.stdout,
                stderr=result.stderr,
                error=result.error,
            )
            for test, result in zip(tests, phase_results, strict=True)
        )
        status = _aggregate_test_status(outcomes)
        return PhaseResult(
            name=phase_name,
            status=status,
            duration_seconds=sum(outcome.duration_seconds for outcome in outcomes),
            stdout=_join_nonempty(outcome.stdout for outcome in outcomes),
            stderr=_join_nonempty(outcome.stderr for outcome in outcomes),
            error=_first_error(outcomes),
            tests_run=len(tests),
            test_outcomes=outcomes,
        )


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


def _final_status(phases: list[PhaseResult], include_hidden: bool) -> PhaseStatus:
    phase_by_name = {phase.name: phase for phase in phases}
    public_phase = phase_by_name.get("public_tests")
    hidden_phase = phase_by_name.get("hidden_tests")
    if public_phase is None or not public_phase.passed:
        return public_phase.status if public_phase is not None else PhaseStatus.RUNTIME_ERROR
    if include_hidden and hidden_phase is not None and not hidden_phase.passed:
        return hidden_phase.status
    return PhaseStatus.PASSED


def _skipped_tests_phase(tests: tuple[TestSpec, ...]) -> PhaseResult:
    outcomes = tuple(
        TestOutcome(
            name=test.name,
            status=PhaseStatus.SKIPPED,
            error="skipped because public tests failed",
        )
        for test in tests
    )
    return PhaseResult(
        name="hidden_tests",
        status=PhaseStatus.SKIPPED,
        error="skipped because public tests failed",
        tests_run=len(tests),
        test_outcomes=outcomes,
    )


def _aggregate_test_status(outcomes: tuple[TestOutcome, ...]) -> PhaseStatus:
    for status in (
        PhaseStatus.TIMEOUT,
        PhaseStatus.SECURITY_REJECTED,
        PhaseStatus.PARSE_ERROR,
        PhaseStatus.RUNTIME_ERROR,
        PhaseStatus.FAILED,
        PhaseStatus.SKIPPED,
    ):
        if any(outcome.status == status for outcome in outcomes):
            return status
    return PhaseStatus.PASSED


def _first_error(outcomes: tuple[TestOutcome, ...]) -> str:
    for outcome in outcomes:
        if not outcome.passed and outcome.error:
            return f"{outcome.name}: {outcome.error}"
    return ""


def _join_nonempty(values) -> str:
    return "\n".join(value for value in values if value)
