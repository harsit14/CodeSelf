"""Correctness-first reward for execution results."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeself.execution import PhaseStatus, TaskRunResult


@dataclass(frozen=True)
class RewardWeights:
    """Weights for reward_v0_correctness."""

    syntax_and_import_success: float = 0.20
    public_test_fraction: float = 0.30
    hidden_test_fraction: float = 0.40
    robustness_test_fraction: float = 0.10


@dataclass(frozen=True)
class RewardPenalties:
    """Penalty values for failure modes."""

    syntax_error: float = -0.25
    unsafe_code: float = -1.00
    timeout: float = -0.50
    runtime_error: float = -0.20
    format_violation: float = -0.10
    flaky_result: float = -0.25


@dataclass(frozen=True)
class RewardConfig:
    """Configuration for correctness reward scoring."""

    name: str = "reward_v0_correctness"
    weights: RewardWeights = field(default_factory=RewardWeights)
    penalties: RewardPenalties = field(default_factory=RewardPenalties)
    normalize_available_weights: bool = True
    clamp_min: float = -1.0
    clamp_max: float = 1.0


@dataclass(frozen=True)
class RewardComponent:
    """One scored component in a reward breakdown."""

    name: str
    value: float
    weight: float
    weighted_value: float
    available: bool = True


@dataclass(frozen=True)
class AppliedPenalty:
    """One applied penalty in a reward breakdown."""

    name: str
    value: float
    reason: str


@dataclass(frozen=True)
class RewardBreakdown:
    """Structured reward output for logging and audit."""

    task_id: str
    reward_name: str
    reward: float
    score_before_penalty: float
    total_penalty: float
    components: tuple[RewardComponent, ...]
    penalties: tuple[AppliedPenalty, ...]
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "reward_name": self.reward_name,
            "reward": self.reward,
            "score_before_penalty": self.score_before_penalty,
            "total_penalty": self.total_penalty,
            "components": [component.__dict__ for component in self.components],
            "penalties": [penalty.__dict__ for penalty in self.penalties],
            "metrics": dict(self.metrics),
        }


def score_correctness(result: TaskRunResult, config: RewardConfig | None = None) -> RewardBreakdown:
    """Score one task run using reward_v0_correctness."""

    reward_config = config or RewardConfig()
    raw_components = _raw_components(result, reward_config.weights)
    components = _normalize_components(raw_components, reward_config)
    score_before_penalty = sum(component.weighted_value for component in components)
    penalties = _applied_penalties(result, reward_config.penalties)
    total_penalty = sum(penalty.value for penalty in penalties)
    reward = _clamp(
        score_before_penalty + total_penalty,
        reward_config.clamp_min,
        reward_config.clamp_max,
    )
    metrics = {
        "passed": result.passed,
        "duration_seconds": result.duration_seconds,
        "phase_count": len(result.phases),
        "timeout": any(phase.status == PhaseStatus.TIMEOUT for phase in result.phases),
        "security_rejected": result.status == PhaseStatus.SECURITY_REJECTED,
    }
    return RewardBreakdown(
        task_id=result.task_id,
        reward_name=reward_config.name,
        reward=reward,
        score_before_penalty=score_before_penalty,
        total_penalty=total_penalty,
        components=components,
        penalties=penalties,
        metrics=metrics,
    )


def _raw_components(result: TaskRunResult, weights: RewardWeights) -> tuple[RewardComponent, ...]:
    syntax = result.phase("syntax")
    import_phase = result.phase("import")
    syntax_and_import = 1.0 if syntax and syntax.passed and import_phase and import_phase.passed else 0.0
    public_phase = result.phase("public_tests")
    hidden_phase = result.phase("hidden_tests")
    robustness_phase = result.phase("robustness_tests")
    return (
        RewardComponent(
            name="syntax_and_import_success",
            value=syntax_and_import,
            weight=weights.syntax_and_import_success,
            weighted_value=0.0,
            available=True,
        ),
        RewardComponent(
            name="public_test_fraction",
            value=_phase_fraction(public_phase),
            weight=weights.public_test_fraction,
            weighted_value=0.0,
            available=_phase_available(public_phase),
        ),
        RewardComponent(
            name="hidden_test_fraction",
            value=_phase_fraction(hidden_phase),
            weight=weights.hidden_test_fraction,
            weighted_value=0.0,
            available=_phase_available(hidden_phase),
        ),
        RewardComponent(
            name="robustness_test_fraction",
            value=_phase_fraction(robustness_phase),
            weight=weights.robustness_test_fraction,
            weighted_value=0.0,
            available=_phase_available(robustness_phase),
        ),
    )


def _normalize_components(
    components: tuple[RewardComponent, ...],
    config: RewardConfig,
) -> tuple[RewardComponent, ...]:
    available_weight = sum(component.weight for component in components if component.available)
    normalized: list[RewardComponent] = []
    for component in components:
        if not component.available:
            normalized.append(component)
            continue
        weight = component.weight
        if config.normalize_available_weights and available_weight > 0:
            weight = component.weight / available_weight
        normalized.append(
            RewardComponent(
                name=component.name,
                value=component.value,
                weight=weight,
                weighted_value=component.value * weight,
                available=component.available,
            )
        )
    return tuple(normalized)


def _phase_fraction(phase) -> float:
    if phase is None or phase.status == PhaseStatus.SKIPPED:
        return 0.0
    return 1.0 if phase.passed else 0.0


def _phase_available(phase) -> bool:
    return phase is not None and phase.status != PhaseStatus.SKIPPED and phase.tests_run > 0


def _applied_penalties(
    result: TaskRunResult,
    penalties: RewardPenalties,
) -> tuple[AppliedPenalty, ...]:
    applied: list[AppliedPenalty] = []
    if result.status == PhaseStatus.PARSE_ERROR:
        applied.append(AppliedPenalty("syntax_error", penalties.syntax_error, "syntax phase failed"))
    if result.status == PhaseStatus.SECURITY_REJECTED:
        applied.append(AppliedPenalty("unsafe_code", penalties.unsafe_code, "security scan rejected code"))
    if any(phase.status == PhaseStatus.TIMEOUT for phase in result.phases):
        applied.append(AppliedPenalty("timeout", penalties.timeout, "one or more phases timed out"))
    if any(phase.status == PhaseStatus.RUNTIME_ERROR for phase in result.phases):
        applied.append(AppliedPenalty("runtime_error", penalties.runtime_error, "runtime error"))
    return tuple(applied)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
