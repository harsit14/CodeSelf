"""Tool abstractions for agentic coding loops."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from codeself.datasets import TaskSpec, TestSpec
from codeself.execution import SandboxedTestRunner
from codeself.rewards import CompositeRewardScorer, RewardScorer


@dataclass(frozen=True)
class ToolCall:
    """One agent tool call."""

    name: str
    arguments: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class ToolResult:
    """Result from one agent tool call."""

    call: ToolCall
    ok: bool
    observation: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.to_dict(),
            "ok": self.ok,
            "observation": self.observation,
            "payload": dict(self.payload),
        }


class AgentToolbox:
    """Public-test and final-submit tools for a coding task."""

    def __init__(
        self,
        runner: SandboxedTestRunner | None = None,
        scorer: RewardScorer | None = None,
    ) -> None:
        self.runner = runner or SandboxedTestRunner()
        self.scorer = scorer or CompositeRewardScorer()

    def run_public_tests(self, task: TaskSpec, code: str) -> ToolResult:
        call = ToolCall("run_public_tests", {"task_id": task.task_id})
        result = self.runner.run(task, code, include_hidden=False)
        phase = result.phase("public_tests")
        ok = result.passed
        observation = _phase_observation(phase, default="public tests were not run")
        return ToolResult(
            call=call,
            ok=ok,
            observation=observation,
            payload={
                "execution": result.to_dict(),
            },
        )

    def run_custom_tests(
        self,
        task: TaskSpec,
        code: str,
        tests: tuple[TestSpec, ...],
    ) -> ToolResult:
        custom_task = TaskSpec(
            task_id=task.task_id,
            source=task.source,
            prompt=task.prompt,
            split=task.split,
            entry_point=task.entry_point,
            starter_code=task.starter_code,
            public_tests=tests,
            hidden_tests=(),
            tags=task.tags,
            resource_limits=task.resource_limits,
            metadata=task.metadata,
        )
        call = ToolCall("run_custom_tests", {"task_id": task.task_id, "tests": len(tests)})
        result = self.runner.run(custom_task, code, include_hidden=False)
        phase = result.phase("public_tests")
        return ToolResult(
            call=call,
            ok=result.passed,
            observation=_phase_observation(phase, default="custom tests were not run"),
            payload={
                "execution": result.to_dict(),
            },
        )

    def submit_final(
        self,
        task: TaskSpec,
        code: str,
        *,
        include_hidden: bool = True,
    ) -> ToolResult:
        call = ToolCall("submit_final", {"task_id": task.task_id})
        result = self.runner.run(task, code, include_hidden=include_hidden)
        reward = self.scorer.score(result, solution_code=code)
        return ToolResult(
            call=call,
            ok=result.passed,
            observation="final submission passed" if result.passed else "final submission failed",
            payload={
                "execution": result.to_dict(),
                "reward": reward.to_dict(),
            },
        )


def infer_simple_revision(task: TaskSpec, code: str, observation: str) -> str:
    """Try a tiny deterministic repair from public assertion patterns.

    This is a smoke-test helper, not a serious program repair system. It lets the
    agentic loop demonstrate the tool/revision contract before model-based
    repair is integrated.
    """

    if not task.entry_point:
        return code
    for test in task.public_tests:
        inferred = _infer_unary_addition(task.entry_point, test.code)
        if inferred is not None:
            return inferred
        inferred_constant = _infer_zero_arg_constant(task.entry_point, test.code)
        if inferred_constant is not None:
            return inferred_constant
    return code


def _infer_unary_addition(entry_point: str, test_code: str) -> str | None:
    pattern = rf"assert\s+{re.escape(entry_point)}\((-?\d+)\)\s*==\s*(-?\d+)"
    match = re.search(pattern, test_code)
    if not match:
        return None
    argument = int(match.group(1))
    expected = int(match.group(2))
    delta = expected - argument
    if delta == 0:
        expression = "x"
    elif delta > 0:
        expression = f"x + {delta}"
    else:
        expression = f"x - {abs(delta)}"
    return f"def {entry_point}(x):\n    return {expression}\n"


def _infer_zero_arg_constant(entry_point: str, test_code: str) -> str | None:
    pattern = rf"assert\s+{re.escape(entry_point)}\(\)\s*==\s*(-?\d+)"
    match = re.search(pattern, test_code)
    if not match:
        return None
    expected = int(match.group(1))
    return f"def {entry_point}():\n    return {expected}\n"


def _phase_observation(phase, *, default: str) -> str:
    if phase is None:
        return default
    if phase.passed:
        return f"{phase.name} passed"
    detail = phase.error or phase.stderr.strip() or "failed"
    return f"{phase.name} {phase.status.value}: {detail}"
