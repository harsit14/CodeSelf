"""Agentic coding loop with public-test self-debugging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codeself.agent.generation import CodeGenerator, GenerationRequest
from codeself.agent.parser import ParsedCompletion, ParseStatus, extract_code
from codeself.agent.prompts import PromptTemplate, get_revision_prompt_template
from codeself.agent.tools import AgentToolbox, ToolResult, infer_simple_revision
from codeself.datasets import TaskSpec


@dataclass(frozen=True)
class AgentLoopConfig:
    """Configuration for self-debug agent runs."""

    max_revisions: int = 1
    seed: int = 20260601
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    revision_max_new_tokens: int | None = None
    revision_temperature: float | None = None
    revision_top_p: float | None = None
    use_rule_based_repair: bool = True
    use_model_revision: bool = False
    revision_prompt_template: str = "self_debug_revision_v1"

    def __post_init__(self) -> None:
        if self.max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.revision_max_new_tokens is not None and self.revision_max_new_tokens <= 0:
            raise ValueError("revision_max_new_tokens must be positive when set")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.revision_temperature is not None and self.revision_temperature < 0:
            raise ValueError("revision_temperature must be non-negative when set")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.revision_top_p is not None and not 0 < self.revision_top_p <= 1:
            raise ValueError("revision_top_p must be in (0, 1] when set")
        get_revision_prompt_template(self.revision_prompt_template)

    @property
    def resolved_revision_max_new_tokens(self) -> int:
        return (
            self.revision_max_new_tokens
            if self.revision_max_new_tokens is not None
            else self.max_new_tokens
        )

    @property
    def resolved_revision_temperature(self) -> float:
        return (
            self.revision_temperature
            if self.revision_temperature is not None
            else self.temperature
        )

    @property
    def resolved_revision_top_p(self) -> float:
        return self.revision_top_p if self.revision_top_p is not None else self.top_p


@dataclass(frozen=True)
class _RevisionCandidate:
    """One proposed self-debug revision."""

    code: str
    parsed_status: str
    observation: str
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True)
class AgentStep:
    """One step in an agent trace."""

    kind: str
    code: str = ""
    parsed_status: str = ""
    observation: str = ""
    tool_result: dict[str, Any] | None = None
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "code": self.code,
            "parsed_status": self.parsed_status,
            "observation": self.observation,
            "tool_result": self.tool_result,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentTrace:
    """Trace for one agentic task attempt."""

    task_id: str
    prompt_template: str
    prompt: str
    raw_completion: str
    initial_code: str
    final_code: str
    steps: tuple[AgentStep, ...]
    final_result: dict[str, Any]
    final_reward: dict[str, Any]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    @property
    def final_passed(self) -> bool:
        return bool(self.final_result.get("passed"))

    @property
    def revision_count(self) -> int:
        return sum(1 for step in self.steps if step.kind == "revision")

    @property
    def tool_call_count(self) -> int:
        return sum(1 for step in self.steps if step.tool_result is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "prompt_template": self.prompt_template,
            "prompt": self.prompt,
            "raw_completion": self.raw_completion,
            "initial_code": self.initial_code,
            "final_code": self.final_code,
            "steps": [step.to_dict() for step in self.steps],
            "final_result": dict(self.final_result),
            "final_reward": dict(self.final_reward),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentTrace":
        steps = tuple(
            AgentStep(
                kind=str(step["kind"]),
                code=str(step.get("code", "")),
                parsed_status=str(step.get("parsed_status", "")),
                observation=str(step.get("observation", "")),
                tool_result=step.get("tool_result"),
                metadata=dict(step.get("metadata", {})),
            )
            for step in value.get("steps", [])
        )
        return cls(
            task_id=str(value["task_id"]),
            prompt_template=str(value["prompt_template"]),
            prompt=str(value["prompt"]),
            raw_completion=str(value["raw_completion"]),
            initial_code=str(value["initial_code"]),
            final_code=str(value["final_code"]),
            steps=steps,
            final_result=dict(value.get("final_result", {})),
            final_reward=dict(value.get("final_reward", {})),
            metadata=dict(value.get("metadata", {})),
        )


class SelfDebugAgentLoop:
    """Generate code, run public tests, revise, then submit final."""

    def __init__(
        self,
        *,
        generator: CodeGenerator,
        prompt_template: PromptTemplate,
        toolbox: AgentToolbox | None = None,
        config: AgentLoopConfig | None = None,
    ) -> None:
        self.generator = generator
        self.prompt_template = prompt_template
        self.toolbox = toolbox or AgentToolbox()
        self.config = config or AgentLoopConfig()
        self.revision_prompt_template = get_revision_prompt_template(
            self.config.revision_prompt_template
        )

    def run_task(
        self,
        task: TaskSpec,
        *,
        sample_index: int = 0,
        include_hidden: bool = True,
    ) -> AgentTrace:
        prompt = self.prompt_template.render(task)
        generation = self.generator.generate(
            GenerationRequest(
                task_id=task.task_id,
                prompt=prompt,
                sample_index=sample_index,
                seed=self.config.seed,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                entry_point=task.entry_point,
            )
        )
        parsed = extract_code(generation.text)
        code = parsed.code
        initial_code = code
        generation_metadata: dict[str, str | int | float | bool] = {
            "backend": generation.backend,
            "model_name": generation.model_name,
            "request_seed": self.config.seed,
            "request_max_new_tokens": self.config.max_new_tokens,
            "request_temperature": self.config.temperature,
            "request_top_p": self.config.top_p,
            **generation.metadata,
        }
        steps: list[AgentStep] = [
            AgentStep(
                kind="generation",
                code=code,
                parsed_status=parsed.status.value,
                metadata=generation_metadata,
            )
        ]

        revisions = 0
        while revisions <= self.config.max_revisions:
            public_result = self.toolbox.run_public_tests(task, code)
            steps.append(_tool_step("run_public_tests", public_result, code=code))
            if public_result.ok or revisions == self.config.max_revisions:
                break

            revision = self._revise(
                task,
                current_code=code,
                observation=public_result.observation,
                revision_index=revisions + 1,
            )
            revisions += 1
            revised = revision.code
            changed = revised != code
            code = revised
            steps.append(
                AgentStep(
                    kind="revision",
                    code=code,
                    parsed_status=revision.parsed_status,
                    observation=revision.observation,
                    metadata={
                        "revision_index": revisions,
                        "changed": changed,
                        **revision.metadata,
                    },
                )
            )
            if not changed:
                break

        final = self.toolbox.submit_final(task, code, include_hidden=include_hidden)
        steps.append(_tool_step("submit_final", final, code=code))
        return AgentTrace(
            task_id=task.task_id,
            prompt_template=self.prompt_template.name,
            prompt=prompt,
            raw_completion=generation.text,
            initial_code=initial_code,
            final_code=code,
            steps=tuple(steps),
            final_result=dict(final.payload["execution"]),
            final_reward=dict(final.payload["reward"]),
            metadata={
                "seed": self.config.seed,
                "include_hidden": include_hidden,
                "max_revisions": self.config.max_revisions,
                "use_rule_based_repair": self.config.use_rule_based_repair,
                "use_model_revision": self.config.use_model_revision,
                "revision_prompt_template": self.config.revision_prompt_template,
                "revision_max_new_tokens": self.config.resolved_revision_max_new_tokens,
                "revision_temperature": self.config.resolved_revision_temperature,
                "revision_top_p": self.config.resolved_revision_top_p,
            },
        )

    def _revise(
        self,
        task: TaskSpec,
        *,
        current_code: str,
        observation: str,
        revision_index: int,
    ) -> _RevisionCandidate:
        if self.config.use_model_revision:
            return self._model_revision(task, current_code, observation, revision_index)
        if self.config.use_rule_based_repair:
            code = infer_simple_revision(task, current_code, observation)
            parsed = _parse_revision(code)
            return _RevisionCandidate(
                code=code,
                parsed_status=parsed.status.value,
                observation="rule-based revision after public-test feedback",
                metadata={"revision_source": "rule_based"},
            )
        parsed = _parse_revision(current_code)
        return _RevisionCandidate(
            code=current_code,
            parsed_status=parsed.status.value,
            observation="no revision strategy enabled",
            metadata={"revision_source": "none"},
        )

    def _model_revision(
        self,
        task: TaskSpec,
        current_code: str,
        observation: str,
        revision_index: int,
    ) -> _RevisionCandidate:
        prompt = self.revision_prompt_template.render(
            task,
            current_code=current_code,
            observation=observation,
            revision_index=revision_index,
        )
        generation = self.generator.generate(
            GenerationRequest(
                task_id=task.task_id,
                prompt=prompt,
                sample_index=revision_index,
                seed=self.config.seed + revision_index,
                max_new_tokens=self.config.resolved_revision_max_new_tokens,
                temperature=self.config.resolved_revision_temperature,
                top_p=self.config.resolved_revision_top_p,
                entry_point=task.entry_point,
            )
        )
        parsed = extract_code(generation.text)
        metadata: dict[str, str | int | float | bool] = {
            "revision_source": "model",
            "revision_prompt_template": self.revision_prompt_template.name,
            "revision_prompt": prompt,
            "raw_revision_completion": generation.text,
            "revision_request_seed": self.config.seed + revision_index,
            "revision_request_max_new_tokens": self.config.resolved_revision_max_new_tokens,
            "revision_request_temperature": self.config.resolved_revision_temperature,
            "revision_request_top_p": self.config.resolved_revision_top_p,
            "backend": generation.backend,
            "model_name": generation.model_name,
            **generation.metadata,
        }
        return _RevisionCandidate(
            code=parsed.code,
            parsed_status=parsed.status.value,
            observation="model revision after public-test feedback",
            metadata=metadata,
        )


@dataclass(frozen=True)
class StrategyAnalysis:
    """Aggregate trace metrics for agentic strategy analysis."""

    trace_count: int
    final_pass_rate: float
    mean_reward: float
    mean_tool_calls: float
    mean_revisions: float
    revision_rate: float
    public_test_pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_count": self.trace_count,
            "final_pass_rate": self.final_pass_rate,
            "mean_reward": self.mean_reward,
            "mean_tool_calls": self.mean_tool_calls,
            "mean_revisions": self.mean_revisions,
            "revision_rate": self.revision_rate,
            "public_test_pass_rate": self.public_test_pass_rate,
        }

    def to_markdown(self) -> str:
        return (
            "# Agentic Strategy Analysis\n\n"
            f"- Traces: {self.trace_count}\n"
            f"- Final pass rate: {self.final_pass_rate:.4f}\n"
            f"- Mean reward: {self.mean_reward:.4f}\n"
            f"- Mean tool calls: {self.mean_tool_calls:.4f}\n"
            f"- Mean revisions: {self.mean_revisions:.4f}\n"
            f"- Revision rate: {self.revision_rate:.4f}\n"
            f"- Public-test pass rate: {self.public_test_pass_rate:.4f}\n"
        )


def run_agentic_tasks(
    tasks: list[TaskSpec],
    *,
    loop: SelfDebugAgentLoop,
    sample_index: int = 0,
) -> list[AgentTrace]:
    return [loop.run_task(task, sample_index=sample_index) for task in tasks]


def analyze_traces(traces: list[AgentTrace]) -> StrategyAnalysis:
    if not traces:
        return StrategyAnalysis(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    public_steps = [
        step for trace in traces for step in trace.steps if step.kind == "run_public_tests"
    ]
    public_passes = [
        step
        for step in public_steps
        if step.tool_result is not None and bool(step.tool_result.get("ok"))
    ]
    return StrategyAnalysis(
        trace_count=len(traces),
        final_pass_rate=_rate(sum(1 for trace in traces if trace.final_passed), len(traces)),
        mean_reward=_mean(float(trace.final_reward.get("reward", 0.0)) for trace in traces),
        mean_tool_calls=_mean(trace.tool_call_count for trace in traces),
        mean_revisions=_mean(trace.revision_count for trace in traces),
        revision_rate=_rate(sum(1 for trace in traces if trace.revision_count > 0), len(traces)),
        public_test_pass_rate=_rate(len(public_passes), len(public_steps)),
    )


def write_agent_traces_jsonl(traces: list[AgentTrace], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), sort_keys=True))
            handle.write("\n")


def read_agent_traces_jsonl(path: str | Path) -> list[AgentTrace]:
    input_path = Path(path)
    traces: list[AgentTrace] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                traces.append(AgentTrace.from_dict(json.loads(stripped)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                message = f"invalid agent trace JSONL at {input_path}:{line_number}"
                raise ValueError(message) from exc
    return traces


def write_strategy_report(analysis: StrategyAnalysis, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".json":
        output_path.write_text(
            json.dumps(analysis.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        output_path.write_text(analysis.to_markdown(), encoding="utf-8")


def _tool_step(kind: str, result: ToolResult, *, code: str) -> AgentStep:
    return AgentStep(
        kind=kind,
        code=code,
        observation=result.observation,
        tool_result=result.to_dict(),
    )


def _parse_revision(code: str) -> ParsedCompletion:
    parsed = extract_code(code)
    if parsed.status == ParseStatus.EMPTY:
        return ParsedCompletion(code, code, ParseStatus.SYNTAX_ERROR, "revision", "empty revision")
    return parsed


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
