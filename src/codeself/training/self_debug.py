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

    def __post_init__(self) -> None:
        if self.max_revisions < 0:
            raise ValueError("max_revisions must be non-negative")
        if self.revision_strategy not in {"rule_based", "model", "none"}:
            raise ValueError("revision_strategy must be rule_based, model, or none")
        if not self.revision_prompt_template.strip():
            raise ValueError("revision_prompt_template must not be empty")
        if not 0 < self.revision_reward_discount <= 1:
            raise ValueError("revision_reward_discount must be in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_revisions": self.max_revisions,
            "revision_strategy": self.revision_strategy,
            "revision_prompt_template": self.revision_prompt_template,
            "revision_reward_discount": self.revision_reward_discount,
        }
