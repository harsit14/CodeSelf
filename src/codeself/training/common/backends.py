"""Training backend registry and availability checks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BackendSpec:
    """Static metadata for one training backend option."""

    name: str
    description: str
    required_packages: tuple[str, ...]
    updates_model_weights: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "required_packages": list(self.required_packages),
            "updates_model_weights": self.updates_model_weights,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class BackendAvailability:
    """Runtime availability for one backend."""

    spec: BackendSpec
    available: bool
    missing_packages: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.spec.to_dict(),
            "available": self.available,
            "missing_packages": list(self.missing_packages),
        }


class TrainerBackend(Protocol):
    """Protocol for future trainable backend implementations."""

    name: str

    def train(self) -> object:
        """Run training and return a backend-specific result."""


BACKEND_SPECS: dict[str, BackendSpec] = {
    "smoke": BackendSpec(
        name="smoke",
        description="Dependency-free diagnostics that do not update weights.",
        required_packages=(),
        updates_model_weights=False,
        notes="Used by CI and local plumbing tests.",
    ),
    "from_scratch": BackendSpec(
        name="from_scratch",
        description="CodeSelf-owned Transformers/PEFT policy-gradient trainer.",
        required_packages=("torch", "transformers", "peft", "accelerate"),
        updates_model_weights=True,
        notes="Primary first implementation target for GRPO/PPO.",
    ),
    "trl": BackendSpec(
        name="trl",
        description="Trainer backed by Hugging Face TRL.",
        required_packages=("torch", "transformers", "peft", "trl"),
        updates_model_weights=True,
        notes="Future backend once the local contracts are stable.",
    ),
    "verl": BackendSpec(
        name="verl",
        description="Trainer backed by verl-style distributed RL infrastructure.",
        required_packages=("torch", "transformers"),
        updates_model_weights=True,
        notes="Future backend for larger-scale experiments.",
    ),
}


def list_backend_specs() -> tuple[BackendSpec, ...]:
    """Return all registered backend specs."""

    return tuple(BACKEND_SPECS[name] for name in sorted(BACKEND_SPECS))


def get_backend_spec(name: str) -> BackendSpec:
    """Return a backend spec by name."""

    try:
        return BACKEND_SPECS[name]
    except KeyError as exc:
        available = ", ".join(sorted(BACKEND_SPECS))
        raise ValueError(f"unknown training backend {name!r}; available: {available}") from exc


def backend_availability(name: str) -> BackendAvailability:
    """Check whether a backend's optional dependencies are importable."""

    spec = get_backend_spec(name)
    missing = tuple(
        package for package in spec.required_packages if importlib.util.find_spec(package) is None
    )
    return BackendAvailability(spec=spec, available=not missing, missing_packages=missing)
