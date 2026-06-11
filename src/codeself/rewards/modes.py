"""Pluggable reward modes built on structured execution results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from codeself.execution import PhaseResult, PhaseStatus, TaskRunResult
from codeself.rewards.correctness import (
    AppliedPenalty,
    RewardBreakdown,
    RewardComponent,
    RewardPenalties,
)


class RewardScorer(Protocol):
    """Interface for reward scorers."""

    def score(self, result: TaskRunResult, *, solution_code: str = "") -> RewardBreakdown:
        """Score one executed candidate."""


@dataclass(frozen=True)
class RewardModeConfig:
    """Configuration for pluggable reward scoring."""

    mode: str = "partial_credit"
    name: str = "reward_partial_credit_v1"
    penalties: RewardPenalties = field(default_factory=RewardPenalties)
    syntax_weight: float = 0.10
    import_weight: float = 0.10
    public_weight: float = 0.40
    hidden_weight: float = 0.40
    timeout_penalty_enabled: bool = True
    exception_penalty_enabled: bool = True
    unsafe_penalty_enabled: bool = True
    parse_penalty_enabled: bool = True
    compile_success_bonus: float = 0.0
    length_penalty_per_1k_chars: float = 0.0
    clamp_min: float = -1.0
    clamp_max: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"binary_all_tests_pass", "fractional_pass_rate", "partial_credit"}:
            raise ValueError("unknown reward mode")
        for value in (
            self.syntax_weight,
            self.import_weight,
            self.public_weight,
            self.hidden_weight,
            self.compile_success_bonus,
            self.length_penalty_per_1k_chars,
        ):
            if value < 0:
                raise ValueError("reward weights and shaping magnitudes must be non-negative")
        if self.clamp_min > self.clamp_max:
            raise ValueError("clamp_min cannot exceed clamp_max")


class ConfigurableRewardScorer:
    """Reward scorer supporting binary, fractional, and partial-credit modes."""

    def __init__(self, config: RewardModeConfig | None = None) -> None:
        self.config = config or RewardModeConfig()

    def score(self, result: TaskRunResult, *, solution_code: str = "") -> RewardBreakdown:
        if self.config.mode == "binary_all_tests_pass":
            components = _binary_components(result)
        elif self.config.mode == "fractional_pass_rate":
            components = _fractional_components(result)
        else:
            components = _partial_credit_components(result, self.config)

        penalties = _configured_penalties(result, self.config)
        shaping = _shaping_components(result, solution_code, self.config)
        all_components = components + shaping
        score_before_penalty = sum(component.weighted_value for component in all_components)
        total_penalty = sum(penalty.value for penalty in penalties)
        reward = _clamp(
            score_before_penalty + total_penalty,
            self.config.clamp_min,
            self.config.clamp_max,
        )
        metrics = {
            "mode": self.config.mode,
            "passed": result.passed,
            "test_pass_fraction": _all_test_fraction(result),
            "public_test_fraction": _phase_test_fraction(result.phase("public_tests")),
            "hidden_test_fraction": _phase_test_fraction(result.phase("hidden_tests")),
            "timeout": any(phase.status == PhaseStatus.TIMEOUT for phase in result.phases),
            "security_rejected": result.status == PhaseStatus.SECURITY_REJECTED,
            "empty_code": not solution_code.strip(),
            "repeated_line_fraction": _repeated_line_fraction(solution_code),
        }
        return RewardBreakdown(
            task_id=result.task_id,
            reward_name=self.config.name,
            reward=reward,
            score_before_penalty=score_before_penalty,
            total_penalty=total_penalty,
            components=all_components,
            penalties=penalties,
            metrics=metrics,
        )


def make_reward_scorer(config: RewardModeConfig | None = None) -> RewardScorer:
    """Create a reward scorer from config."""

    return ConfigurableRewardScorer(config)


def _binary_components(result: TaskRunResult) -> tuple[RewardComponent, ...]:
    value = 1.0 if result.passed else 0.0
    return (
        RewardComponent(
            name="all_tests_pass",
            value=value,
            weight=1.0,
            weighted_value=value,
            available=True,
        ),
    )


def _fractional_components(result: TaskRunResult) -> tuple[RewardComponent, ...]:
    value = _all_test_fraction(result)
    return (
        RewardComponent(
            name="test_pass_fraction",
            value=value,
            weight=1.0,
            weighted_value=value,
            available=True,
        ),
    )


def _partial_credit_components(
    result: TaskRunResult,
    config: RewardModeConfig,
) -> tuple[RewardComponent, ...]:
    syntax = result.phase("syntax")
    import_phase = result.phase("import")
    public_phase = result.phase("public_tests")
    hidden_phase = result.phase("hidden_tests")
    return (
        _component(
            "syntax_success",
            1.0 if syntax and syntax.passed else 0.0,
            config.syntax_weight,
            available=syntax is not None,
        ),
        _component(
            "import_success",
            1.0 if import_phase and import_phase.passed else 0.0,
            config.import_weight,
            available=import_phase is not None,
        ),
        _component(
            "public_test_fraction",
            _phase_test_fraction(public_phase),
            config.public_weight,
            available=_phase_available(public_phase),
        ),
        _component(
            "hidden_test_fraction",
            _phase_test_fraction(hidden_phase),
            config.hidden_weight,
            available=_phase_available(hidden_phase),
        ),
    )


def _shaping_components(
    result: TaskRunResult,
    solution_code: str,
    config: RewardModeConfig,
) -> tuple[RewardComponent, ...]:
    components: list[RewardComponent] = []
    if config.compile_success_bonus:
        syntax = result.phase("syntax")
        value = 1.0 if syntax and syntax.passed else 0.0
        components.append(
            _component("compile_success_bonus", value, config.compile_success_bonus, available=True)
        )
    if config.length_penalty_per_1k_chars and solution_code:
        value = -len(solution_code) / 1000
        components.append(
            _component(
                "length_penalty",
                value,
                config.length_penalty_per_1k_chars,
                available=True,
            )
        )
    return tuple(components)


def _configured_penalties(
    result: TaskRunResult,
    config: RewardModeConfig,
) -> tuple[AppliedPenalty, ...]:
    penalties: list[AppliedPenalty] = []
    if config.parse_penalty_enabled and result.status == PhaseStatus.PARSE_ERROR:
        penalties.append(
            AppliedPenalty("syntax_error", config.penalties.syntax_error, "syntax phase failed")
        )
    if config.unsafe_penalty_enabled and result.status == PhaseStatus.SECURITY_REJECTED:
        penalties.append(
            AppliedPenalty(
                "unsafe_code",
                config.penalties.unsafe_code,
                "security scan rejected code",
            )
        )
    if config.timeout_penalty_enabled and any(
        phase.status == PhaseStatus.TIMEOUT for phase in result.phases
    ):
        penalties.append(
            AppliedPenalty("timeout", config.penalties.timeout, "one or more phases timed out")
        )
    if config.exception_penalty_enabled and any(
        phase.status == PhaseStatus.RUNTIME_ERROR for phase in result.phases
    ):
        penalties.append(
            AppliedPenalty("runtime_error", config.penalties.runtime_error, "runtime error")
        )
    return tuple(penalties)


def _all_test_fraction(result: TaskRunResult) -> float:
    phases = [result.phase("public_tests"), result.phase("hidden_tests")]
    outcomes = [
        outcome
        for phase in phases
        if phase is not None
        for outcome in phase.test_outcomes
        if outcome.status != PhaseStatus.SKIPPED
    ]
    if outcomes:
        return sum(1 for outcome in outcomes if outcome.passed) / len(outcomes)
    available_phases = [phase for phase in phases if _phase_available(phase)]
    if not available_phases:
        return 1.0 if result.passed else 0.0
    return sum(_phase_test_fraction(phase) for phase in available_phases) / len(available_phases)


def _phase_test_fraction(phase: PhaseResult | None) -> float:
    if phase is None or phase.status == PhaseStatus.SKIPPED:
        return 0.0
    outcomes = [outcome for outcome in phase.test_outcomes if outcome.status != PhaseStatus.SKIPPED]
    if outcomes:
        return sum(1 for outcome in outcomes if outcome.passed) / len(outcomes)
    return 1.0 if phase.passed else 0.0


def _phase_available(phase: PhaseResult | None) -> bool:
    return phase is not None and phase.status != PhaseStatus.SKIPPED and phase.tests_run > 0


def _component(
    name: str,
    value: float,
    weight: float,
    *,
    available: bool,
) -> RewardComponent:
    return RewardComponent(
        name=name,
        value=value,
        weight=weight,
        weighted_value=value * weight if available else 0.0,
        available=available,
    )


def _repeated_line_fraction(code: str) -> float:
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return 1.0 - (len(set(lines)) / len(lines))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
