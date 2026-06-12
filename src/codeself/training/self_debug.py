"""Self-debug rollout collection settings for training cycles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfDebugCollectionConfig:
    """Settings for collecting self-debug rollouts inside training loops."""

    max_revisions: int = 1
    revision_strategy: str = "rule_based"
    revision_prompt_template: str = "self_debug_revision_v1"
    revision_reward_discount: float = 1.0
    revision_max_new_tokens: int | None = None
    revision_temperature: float | None = None
    revision_top_p: float | None = None

    def __post_init__(self) -> None:
        if self.max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        if self.revision_strategy not in {"rule_based", "model", "none"}:
            raise ValueError("revision_strategy must be rule_based, model, or none")
        if not self.revision_prompt_template.strip():
            raise ValueError("revision_prompt_template must not be empty")
        if not 0 < self.revision_reward_discount <= 1:
            raise ValueError("revision_reward_discount must be in (0, 1]")
        if self.revision_max_new_tokens is not None and self.revision_max_new_tokens <= 0:
            raise ValueError("revision_max_new_tokens must be positive when set")
        if self.revision_temperature is not None and self.revision_temperature < 0:
            raise ValueError("revision_temperature must be non-negative when set")
        if self.revision_top_p is not None and not 0 < self.revision_top_p <= 1:
            raise ValueError("revision_top_p must be in (0, 1] when set")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_revisions": self.max_revisions,
            "revision_strategy": self.revision_strategy,
            "revision_prompt_template": self.revision_prompt_template,
            "revision_reward_discount": self.revision_reward_discount,
            "revision_max_new_tokens": self.revision_max_new_tokens,
            "revision_temperature": self.revision_temperature,
            "revision_top_p": self.revision_top_p,
        }
