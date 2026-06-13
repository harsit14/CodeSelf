"""Pluggable experiment trackers.

A single ``ExperimentTracker`` interface with four backends:

- ``wandb``       — Weights & Biases (optional dependency).
- ``tensorboard`` — TensorBoard SummaryWriter (optional dependency).
- ``jsonl``       — always-available local JSONL sink (the default fallback).
- ``null``        — discards everything (for tests / disabled tracking).

The point is that switching backends is a config change, not a code change,
and the JSONL fallback keeps tracking working with zero third-party deps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol


class ExperimentTracker(Protocol):
    """Minimal metric-logging surface shared by all backends."""

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        """Log a dict of scalar metrics at a training step."""

    def log_config(self, config: dict[str, Any]) -> None:
        """Record run configuration / hyperparameters."""

    def finish(self) -> None:
        """Flush and close the tracker."""


class NullTracker:
    """Tracker that discards all metrics."""

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        return None

    def log_config(self, config: dict[str, Any]) -> None:
        return None

    def finish(self) -> None:
        return None


class JsonlTracker:
    """Append-only JSONL tracker — the dependency-free default."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        record = {"step": step, **{k: _to_scalar(v) for k, v in metrics.items()}}
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def log_config(self, config: dict[str, Any]) -> None:
        self._handle.write(json.dumps({"config": config}, sort_keys=True) + "\n")
        self._handle.flush()

    def finish(self) -> None:
        self._handle.close()


class WandbTracker:
    """Weights & Biases tracker (optional dependency)."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        try:
            import wandb
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WandbTracker requires the optional `wandb` package "
                "(pip install -e '.[tracking]')"
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config or {})

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        self._run.log({k: _to_scalar(v) for k, v in metrics.items()}, step=step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._run.config.update(config, allow_val_change=True)

    def finish(self) -> None:
        self._run.finish()


class TensorBoardTracker:
    """TensorBoard tracker (optional dependency)."""

    def __init__(self, log_dir: str | Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            try:
                from tensorboardX import SummaryWriter  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "TensorBoardTracker requires torch.utils.tensorboard or tensorboardX "
                    "(pip install -e '.[tracking]')"
                ) from exc
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: dict[str, float], *, step: int) -> None:
        for key, value in metrics.items():
            self._writer.add_scalar(key, _to_scalar(value), step)

    def log_config(self, config: dict[str, Any]) -> None:
        self._writer.add_text("config", json.dumps(config, indent=2, sort_keys=True))

    def finish(self) -> None:
        self._writer.flush()
        self._writer.close()


def make_tracker(
    backend: str,
    *,
    project: str = "codeself",
    run_name: str | None = None,
    log_dir: str | Path | None = None,
    jsonl_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> ExperimentTracker:
    """Create a tracker by backend name.

    ``wandb`` and ``tensorboard`` fall back to ``jsonl`` if their optional
    dependency is missing, so a missing package never crashes a run.
    """

    backend = backend.lower()
    if backend in {"none", "null", "off"}:
        return NullTracker()
    if backend == "jsonl":
        return JsonlTracker(jsonl_path or "metrics_tracker.jsonl")
    if backend == "wandb":
        try:
            return WandbTracker(project=project, run_name=run_name, config=config)
        except RuntimeError:
            return JsonlTracker(jsonl_path or "metrics_tracker.jsonl")
    if backend == "tensorboard":
        try:
            return TensorBoardTracker(log_dir or "tensorboard")
        except RuntimeError:
            return JsonlTracker(jsonl_path or "metrics_tracker.jsonl")
    raise ValueError(f"unknown tracker backend: {backend}")


def _to_scalar(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
