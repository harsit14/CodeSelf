"""Self-debug rollout generation backed by agent traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from codeself.agent.generation import CodeGenerator
from codeself.agent.loop import (
    AgentLoopConfig,
    AgentStep,
    AgentTrace,
    SelfDebugAgentLoop,
    StrategyAnalysis,
    analyze_traces,
)
from codeself.agent.parser import extract_code
from codeself.agent.prompts import PromptTemplate
from codeself.agent.rollouts import RolloutRecord
from codeself.agent.tools import AgentToolbox
from codeself.datasets import TaskSpec
from codeself.execution import SandboxedTestRunner
from codeself.rewards import RewardScorer


@dataclass(frozen=True)
class SelfDebugRolloutConfig:
    """Configuration for turning self-debug traces into rollout records."""

    seed: int = 20260601
    max_revisions: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    include_hidden: bool = True
    use_rule_based_repair: bool = True
    revision_strategy: str = "rule_based"
    revision_prompt_template: str = "self_debug_revision_v1"
    revision_reward_discount: float = 1.0

    def __post_init__(self) -> None:
        if self.revision_strategy not in {"rule_based", "model", "none"}:
            raise ValueError("revision_strategy must be rule_based, model, or none")
        if self.max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 < self.revision_reward_discount <= 1:
            raise ValueError("revision_reward_discount must be in (0, 1]")

    def to_agent_loop_config(self) -> AgentLoopConfig:
        return AgentLoopConfig(
            max_revisions=self.max_revisions,
            seed=self.seed,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            use_rule_based_repair=(
                self.use_rule_based_repair and self.revision_strategy == "rule_based"
            ),
            use_model_revision=self.revision_strategy == "model",
            revision_prompt_template=self.revision_prompt_template,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "max_revisions": self.max_revisions,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "include_hidden": self.include_hidden,
            "use_rule_based_repair": self.use_rule_based_repair,
            "revision_strategy": self.revision_strategy,
            "revision_prompt_template": self.revision_prompt_template,
            "revision_reward_discount": self.revision_reward_discount,
        }


@dataclass(frozen=True)
class SelfDebugRolloutResult:
    """Training-ready rollouts plus trace artifacts for one self-debug run."""

    records: tuple[RolloutRecord, ...]
    traces: tuple[AgentTrace, ...]
    config: SelfDebugRolloutConfig

    @property
    def analysis(self) -> StrategyAnalysis:
        return analyze_traces(list(self.traces))

    def to_dict(self) -> dict[str, object]:
        return {
            "records": [record.to_dict() for record in self.records],
            "traces": [trace.to_dict() for trace in self.traces],
            "analysis": self.analysis.to_dict(),
            "config": self.config.to_dict(),
        }


def generate_self_debug_rollouts(
    tasks: Iterable[TaskSpec],
    *,
    generator: CodeGenerator,
    prompt_template: PromptTemplate,
    samples_per_task: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    include_hidden: bool,
    max_revisions: int = 1,
    use_rule_based_repair: bool = True,
    revision_strategy: str = "rule_based",
    revision_prompt_template: str = "self_debug_revision_v1",
    revision_reward_discount: float = 1.0,
    scorer: RewardScorer | None = None,
    runner: SandboxedTestRunner | None = None,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> SelfDebugRolloutResult:
    """Run self-debug attempts and return standard rollout records."""

    if samples_per_task <= 0:
        raise ValueError("samples_per_task must be positive")
    effective_revision_strategy = revision_strategy
    if not use_rule_based_repair and revision_strategy == "rule_based":
        effective_revision_strategy = "none"
    config = SelfDebugRolloutConfig(
        seed=seed,
        max_revisions=max_revisions,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        include_hidden=include_hidden,
        use_rule_based_repair=use_rule_based_repair,
        revision_strategy=effective_revision_strategy,
        revision_prompt_template=revision_prompt_template,
        revision_reward_discount=revision_reward_discount,
    )
    toolbox = AgentToolbox(runner=runner, scorer=scorer)
    loop = SelfDebugAgentLoop(
        generator=generator,
        prompt_template=prompt_template,
        toolbox=toolbox,
        config=config.to_agent_loop_config(),
    )
    records: list[RolloutRecord] = []
    traces: list[AgentTrace] = []
    for task in tasks:
        for sample_index in range(samples_per_task):
            trace = loop.run_task(
                task,
                sample_index=sample_index,
                include_hidden=config.include_hidden,
            )
            traces.append(trace)
            records.append(
                rollout_record_from_trace(
                    trace,
                    sample_index=sample_index,
                    revision_reward_discount=config.revision_reward_discount,
                    metadata=metadata,
                )
            )
    return SelfDebugRolloutResult(
        records=tuple(records),
        traces=tuple(traces),
        config=config,
    )


def rollout_record_from_trace(
    trace: AgentTrace,
    *,
    sample_index: int = 0,
    revision_reward_discount: float = 1.0,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> RolloutRecord:
    """Convert the final attempt in an agent trace into a rollout record."""

    if not 0 < revision_reward_discount <= 1:
        raise ValueError("revision_reward_discount must be in (0, 1]")
    parsed = extract_code(trace.final_code)
    generation_step = _first_step(trace, "generation")
    generation_metadata = dict(generation_step.metadata) if generation_step else {}
    backend = str(generation_metadata.pop("backend", "self_debug"))
    model_name = str(generation_metadata.pop("model_name", "unknown"))
    initial_parse_status = generation_step.parsed_status if generation_step else ""
    reward = _discounted_reward(
        trace.final_reward,
        revision_count=trace.revision_count,
        revision_reward_discount=revision_reward_discount,
    )
    return RolloutRecord(
        task_id=trace.task_id,
        sample_index=sample_index,
        prompt_template=trace.prompt_template,
        prompt=trace.prompt,
        raw_completion=trace.final_code,
        parsed=parsed,
        backend=backend,
        model_name=model_name,
        generation_metadata=_scalar_metadata(generation_metadata),
        execution=dict(trace.final_result),
        reward=reward,
        metadata={
            "rollout_mode": "self_debug",
            "seed": int(trace.metadata.get("seed", 0)),
            "include_hidden": bool(trace.metadata.get("include_hidden", True)),
            "max_revisions": int(trace.metadata.get("max_revisions", 0)),
            "use_rule_based_repair": bool(
                trace.metadata.get("use_rule_based_repair", True)
            ),
            "use_model_revision": bool(trace.metadata.get("use_model_revision", False)),
            "revision_strategy": _revision_strategy(trace),
            "revision_prompt_template": str(
                trace.metadata.get("revision_prompt_template", "")
            ),
            "revision_count": trace.revision_count,
            "tool_call_count": trace.tool_call_count,
            "public_test_passed": _last_tool_ok(trace, "run_public_tests"),
            "final_passed": trace.final_passed,
            "initial_parse_status": initial_parse_status,
            "revision_reward_discount": revision_reward_discount,
            **(metadata or {}),
        },
    )


def _discounted_reward(
    reward: dict[str, object],
    *,
    revision_count: int,
    revision_reward_discount: float,
) -> dict[str, object]:
    payload = dict(reward)
    raw_reward = _float(payload.get("reward"))
    discount_factor = revision_reward_discount**revision_count
    discounted_reward = raw_reward * discount_factor if raw_reward > 0 else raw_reward
    payload["reward"] = discounted_reward
    metrics = dict(payload.get("metrics", {})) if isinstance(payload.get("metrics"), dict) else {}
    metrics.update(
        {
            "self_debug_undiscounted_final_reward": raw_reward,
            "self_debug_revision_count": revision_count,
            "self_debug_revision_reward_discount": revision_reward_discount,
            "self_debug_discount_factor": discount_factor,
        }
    )
    payload["metrics"] = metrics
    return payload


def _first_step(trace: AgentTrace, kind: str) -> AgentStep | None:
    return next((step for step in trace.steps if step.kind == kind), None)


def _last_tool_ok(trace: AgentTrace, kind: str) -> bool:
    steps = [step for step in trace.steps if step.kind == kind and step.tool_result]
    if not steps:
        return False
    return bool(steps[-1].tool_result.get("ok"))


def _revision_strategy(trace: AgentTrace) -> str:
    if bool(trace.metadata.get("use_model_revision", False)):
        return "model"
    if bool(trace.metadata.get("use_rule_based_repair", True)):
        return "rule_based"
    return "none"


def _scalar_metadata(
    metadata: dict[str, object],
) -> dict[str, str | int | float | bool]:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(key, str) and isinstance(value, (str, int, float, bool))
    }


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0
