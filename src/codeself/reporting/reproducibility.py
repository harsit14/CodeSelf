"""Reproducibility manifest and archive helpers."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from codeself.config import ConfigLoadError, config_get, load_config_file


DEFAULT_INCLUDE_PATTERNS = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "methodology.md",
    "pyproject.toml",
    "configs/**",
    "docker/**",
    "docs/**",
    "scripts/*.py",
    "src/**/*.py",
    "tests/**/*.py",
)

DEFAULT_EXCLUDE_PATTERNS = (
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "venv/**",
    "blog/**",
    "data/raw/**",
    "data/processed/**",
    "data/splits/**",
    "outputs/checkpoints/**",
    "outputs/evals/**",
    "outputs/figures/**",
    "outputs/rollouts/**",
    "outputs/reports/*",
    "plan.md",
    "progress.md",
)

GROUP_EXCLUDE_PATTERNS = (
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
    ".venv/**",
    "venv/**",
)

CONFIG_HASH_PATTERNS = (
    "configs/**/*.json",
    "configs/**/*.jsonl",
    "configs/**/*.yaml",
    "configs/**/*.yml",
)

DATASET_VERSION_PATTERNS = (
    "configs/datasets/**",
    "data/splits/**",
)

ENVIRONMENT_LOCKFILE_PATTERNS = (
    "pyproject.toml",
    "requirements*.txt",
    "requirements/**/*.txt",
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "conda*.yml",
    "environment*.yml",
    "docker/**",
)

CHECKPOINT_MANIFEST_PATTERNS = (
    "outputs/checkpoints/**/checkpoint*.json",
    "outputs/checkpoints/**/checkpoint_manifest.json",
    "artifacts/**/checkpoint*.json",
)


@dataclass(frozen=True)
class FileArtifact:
    """One file captured in the reproducibility manifest."""

    path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class CommandSpec:
    """Command that should reproduce a core project stage."""

    name: str
    command: str
    purpose: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "command": self.command, "purpose": self.purpose}


@dataclass(frozen=True)
class ModelReference:
    """Model/tokenizer reference extracted from an experiment config."""

    source_path: str
    config_path: str
    role: str
    name: str
    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "source_path": self.source_path,
            "config_path": self.config_path,
            "role": self.role,
            "name": self.name,
            "tokenizer": self.tokenizer,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True)
class CheckpointArtifactReference:
    """Checkpoint artifact checksum copied from a checkpoint manifest."""

    source_path: str
    kind: str
    path: str
    bytes: int | None
    sha256: str

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "source_path": self.source_path,
            "kind": self.kind,
            "path": self.path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ReproducibilityManifest:
    """Public metadata needed to reproduce a CodeSelf run."""

    generated_at_utc: str
    project_name: str
    package_version: str
    python_version: str
    platform: str
    git_revision: str | None
    git_status_short: tuple[str, ...]
    artifact_count: int
    artifacts: tuple[FileArtifact, ...]
    config_artifacts: tuple[FileArtifact, ...]
    dataset_artifacts: tuple[FileArtifact, ...]
    environment_artifacts: tuple[FileArtifact, ...]
    checkpoint_manifests: tuple[FileArtifact, ...]
    checkpoint_artifacts: tuple[CheckpointArtifactReference, ...]
    model_references: tuple[ModelReference, ...]
    commands: tuple[CommandSpec, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "project_name": self.project_name,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "git_revision": self.git_revision,
            "git_status_short": list(self.git_status_short),
            "artifact_count": self.artifact_count,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "config_artifact_count": len(self.config_artifacts),
            "config_artifacts": [artifact.to_dict() for artifact in self.config_artifacts],
            "dataset_artifact_count": len(self.dataset_artifacts),
            "dataset_artifacts": [
                artifact.to_dict() for artifact in self.dataset_artifacts
            ],
            "environment_artifact_count": len(self.environment_artifacts),
            "environment_artifacts": [
                artifact.to_dict() for artifact in self.environment_artifacts
            ],
            "checkpoint_manifest_count": len(self.checkpoint_manifests),
            "checkpoint_manifests": [
                artifact.to_dict() for artifact in self.checkpoint_manifests
            ],
            "checkpoint_artifact_count": len(self.checkpoint_artifacts),
            "checkpoint_artifacts": [
                artifact.to_dict() for artifact in self.checkpoint_artifacts
            ],
            "model_reference_count": len(self.model_references),
            "model_references": [
                reference.to_dict() for reference in self.model_references
            ],
            "commands": [command.to_dict() for command in self.commands],
            "notes": list(self.notes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_markdown(self) -> str:
        status = "\n".join(f"- `{line}`" for line in self.git_status_short) or "- clean"
        commands = "\n".join(
            f"### {command.name}\n\n{command.purpose}\n\n```bash\n{command.command}\n```"
            for command in self.commands
        )
        artifacts = "\n".join(
            f"- `{artifact.path}` ({artifact.bytes} bytes, sha256 `{artifact.sha256}`)"
            for artifact in self.artifacts
        )
        configs = _artifact_markdown(self.config_artifacts)
        datasets = _artifact_markdown(self.dataset_artifacts)
        environment = _artifact_markdown(self.environment_artifacts)
        checkpoints = _checkpoint_markdown(
            self.checkpoint_manifests,
            self.checkpoint_artifacts,
        )
        models = _model_reference_markdown(self.model_references)
        notes = "\n".join(f"- {note}" for note in self.notes)
        return (
            "# CodeSelf Reproducibility Manifest\n\n"
            f"- Generated: {self.generated_at_utc}\n"
            f"- Package version: {self.package_version}\n"
            f"- Python: {self.python_version}\n"
            f"- Platform: {self.platform}\n"
            f"- Git revision: {self.git_revision or 'unavailable'}\n"
            f"- Artifact count: {self.artifact_count}\n\n"
            "## Git Status\n\n"
            f"{status}\n\n"
            "## Reproduction Commands\n\n"
            f"{commands}\n\n"
            "## Artifacts\n\n"
            f"{artifacts}\n\n"
            "## Experiment Config Hashes\n\n"
            f"{configs}\n\n"
            "## Dataset And Split Artifacts\n\n"
            f"{datasets}\n\n"
            "## Environment Lockfiles\n\n"
            f"{environment}\n\n"
            "## Model References\n\n"
            f"{models}\n\n"
            "## Checkpoint Manifests\n\n"
            f"{checkpoints}\n\n"
            "## Notes\n\n"
            f"{notes}\n"
        )


def default_reproduction_commands() -> tuple[CommandSpec, ...]:
    """Return the core commands a reviewer can run locally."""

    return (
        CommandSpec(
            name="Validate example task schema",
            command="python3 scripts/validate_task_schema.py configs/datasets/tasks.example.jsonl",
            purpose="Confirms the canonical task format and split labels are parseable.",
        ),
        CommandSpec(
            name="Run unit tests",
            command="python3 -m unittest discover -s tests",
            purpose=(
                "Exercises dataset loading, sandbox execution, rewards, rollouts, "
                "training smoke tests, and reports."
            ),
        ),
        CommandSpec(
            name="Generate mock rollouts",
            command=(
                "python3 scripts/run_rollouts.py --tasks configs/datasets/tasks.example.jsonl "
                "--output outputs/rollouts/mock_smoke.jsonl --backend mock --samples-per-task 4"
            ),
            purpose="Creates one JSONL rollout file using the dependency-free mock backend.",
        ),
        CommandSpec(
            name="Evaluate rollouts",
            command=(
                "python3 scripts/evaluate_rollouts.py --rollouts "
                "outputs/rollouts/mock_smoke.jsonl "
                "--output outputs/reports/mock_baseline.md "
                "--power-output outputs/reports/mock_power.json"
            ),
            purpose=(
                "Summarizes pass@k, parser failures, reward statistics, and "
                "power-planning diagnostics."
            ),
        ),
        CommandSpec(
            name="Run GRPO smoke diagnostics",
            command=(
                "python3 scripts/train_grpo_smoke.py --tasks configs/datasets/tasks.example.jsonl "
                "--output-dir outputs/checkpoints/grpo_smoke_example "
                "--backend mock --group-size 4 --max-steps 3"
            ),
            purpose=(
                "Validates grouped reward advantages and checkpoint/report plumbing "
                "without model fine-tuning."
            ),
        ),
        CommandSpec(
            name="Run PPO smoke diagnostics",
            command=(
                "python3 scripts/train_ppo_smoke.py --tasks configs/datasets/tasks.example.jsonl "
                "--output-dir outputs/checkpoints/ppo_smoke_example "
                "--backend mock --samples-per-task 4 --max-steps 3"
            ),
            purpose="Validates value-baseline PPO diagnostics and matched-budget report plumbing.",
        ),
        CommandSpec(
            name="Compare PPO and GRPO smoke outputs",
            command=(
                "python3 scripts/compare_training_smoke.py "
                "--grpo-metrics outputs/checkpoints/grpo_smoke_example/metrics.jsonl "
                "--ppo-metrics outputs/checkpoints/ppo_smoke_example/metrics.jsonl "
                "--output outputs/reports/ppo_vs_grpo_report.md"
            ),
            purpose="Writes the algorithm comparison table used by the methodology package.",
        ),
        CommandSpec(
            name="Create reproducibility manifest",
            command=(
                "python3 scripts/make_reproducibility_manifest.py "
                "--output outputs/reports/reproducibility_manifest.json "
                "--markdown-output outputs/reports/reproducibility_manifest.md "
                "--archive outputs/reports/codeself_reproducibility.tar.gz"
            ),
            purpose=(
                "Records artifact checksums, environment metadata, git status, "
                "and rerun commands."
            ),
        ),
    )


def sha256_file(path: str | Path) -> str:
    """Compute a SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(
    root: str | Path,
    include_patterns: Iterable[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Iterable[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> tuple[FileArtifact, ...]:
    """Collect files matching the public reproducibility surface."""

    root_path = Path(root).resolve()
    include = tuple(include_patterns)
    exclude = tuple(exclude_patterns)
    artifacts: list[FileArtifact] = []
    for path in sorted(root_path.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root_path).as_posix()
        if not _matches_any(relative, include):
            continue
        if _matches_any(relative, exclude):
            continue
        artifacts.append(
            FileArtifact(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path))
        )
    return tuple(artifacts)


def build_reproducibility_manifest(
    root: str | Path,
    generated_at_utc: str | None = None,
) -> ReproducibilityManifest:
    """Build a manifest for the current repository state."""

    root_path = Path(root).resolve()
    generated_at = (
        generated_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    artifacts = collect_artifacts(root_path)
    config_artifacts = _collect_file_group(root_path, CONFIG_HASH_PATTERNS)
    dataset_artifacts = _collect_file_group(root_path, DATASET_VERSION_PATTERNS)
    environment_artifacts = _collect_file_group(root_path, ENVIRONMENT_LOCKFILE_PATTERNS)
    checkpoint_manifests = _collect_file_group(root_path, CHECKPOINT_MANIFEST_PATTERNS)
    return ReproducibilityManifest(
        generated_at_utc=generated_at,
        project_name="CodeSelf",
        package_version=_package_version(root_path),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        git_revision=_git_output(root_path, "rev-parse", "HEAD"),
        git_status_short=tuple(_git_status(root_path)),
        artifact_count=len(artifacts),
        artifacts=artifacts,
        config_artifacts=config_artifacts,
        dataset_artifacts=dataset_artifacts,
        environment_artifacts=environment_artifacts,
        checkpoint_manifests=checkpoint_manifests,
        checkpoint_artifacts=_checkpoint_artifact_references(root_path, checkpoint_manifests),
        model_references=_model_references(root_path, config_artifacts),
        commands=default_reproduction_commands(),
        notes=(
            (
                "Smoke commands use dependency-free mock generation; they do not "
                "fine-tune model weights."
            ),
            (
                "Model/tokenizer references are parsed from configs; pin exact remote "
                "revisions before paper-grade runs."
            ),
            (
                "Checkpoint artifact hashes are copied from checkpoint manifests when "
                "those manifests exist."
            ),
            (
                "Private hidden tests should be archived separately and never embedded "
                "in prompts or public rollouts."
            ),
        ),
    )


def write_manifest_json(manifest: ReproducibilityManifest, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_json(), encoding="utf-8")


def write_manifest_markdown(manifest: ReproducibilityManifest, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(manifest.to_markdown(), encoding="utf-8")


def write_reproducibility_archive(
    manifest: ReproducibilityManifest,
    root: str | Path,
    path: str | Path,
) -> None:
    """Write a tar.gz archive containing manifest metadata and public artifacts."""

    root_path = Path(root).resolve()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        manifest_bytes = manifest.to_json().encode("utf-8")
        manifest_info = tarfile.TarInfo("codeself_reproducibility/manifest.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = 0
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))
        for artifact in manifest.artifacts:
            source = root_path / artifact.path
            if source.exists():
                archive.add(source, arcname=f"codeself_reproducibility/{artifact.path}")


def _collect_file_group(
    root: Path,
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str] = GROUP_EXCLUDE_PATTERNS,
) -> tuple[FileArtifact, ...]:
    artifacts: list[FileArtifact] = []
    include = tuple(include_patterns)
    exclude = tuple(exclude_patterns)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if _matches_any(relative, exclude):
            continue
        if not _matches_any(relative, include):
            continue
        artifacts.append(
            FileArtifact(path=relative, bytes=path.stat().st_size, sha256=sha256_file(path))
        )
    return tuple(artifacts)


def _model_references(
    root: Path,
    config_artifacts: tuple[FileArtifact, ...],
) -> tuple[ModelReference, ...]:
    references: list[ModelReference] = []
    seen: set[tuple[str, str, str]] = set()
    for artifact in config_artifacts:
        path = root / artifact.path
        try:
            config = load_config_file(path)
        except (ConfigLoadError, OSError, UnicodeDecodeError):
            continue
        for reference in _model_references_for_config(artifact.path, config):
            identity = (reference.source_path, reference.config_path, reference.role)
            if identity in seen:
                continue
            seen.add(identity)
            references.append(reference)
    return tuple(references)


def _model_references_for_config(
    source_path: str,
    config: dict[str, Any],
) -> tuple[ModelReference, ...]:
    references: list[ModelReference] = []
    model = config_get(config, "model")
    if isinstance(model, dict):
        role = "policy" if _looks_like_training_model_block(model) else "model"
        references.extend(_reference_from_mapping(source_path, "model", role, model))
        value_model = model.get("value_model")
        if isinstance(value_model, dict):
            references.extend(
                _reference_from_mapping(
                    source_path,
                    "model.value_model",
                    "value_model",
                    value_model,
                )
            )
    generation = config_get(config, "generation")
    if isinstance(generation, dict):
        references.extend(
            _reference_from_mapping(source_path, "generation", "generation", generation)
        )
    return tuple(references)


def _reference_from_mapping(
    source_path: str,
    config_path: str,
    role: str,
    value: dict[str, Any],
) -> tuple[ModelReference, ...]:
    name = value.get("name") or value.get("model_name_or_path") or value.get("model")
    tokenizer = value.get("tokenizer")
    if name is None and tokenizer is None:
        return ()
    return (
        ModelReference(
            source_path=source_path,
            config_path=config_path,
            role=role,
            name=str(name or tokenizer),
            tokenizer=None if tokenizer is None else str(tokenizer),
            revision=_optional_str(value.get("revision")),
            tokenizer_revision=_optional_str(value.get("tokenizer_revision")),
        ),
    )


def _looks_like_training_model_block(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "backend",
            "full_finetune",
            "use_lora",
            "value_model",
            "old_policy",
            "reference",
        )
    )


def _checkpoint_artifact_references(
    root: Path,
    checkpoint_manifests: tuple[FileArtifact, ...],
) -> tuple[CheckpointArtifactReference, ...]:
    references: list[CheckpointArtifactReference] = []
    for artifact in checkpoint_manifests:
        path = root / artifact.path
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for item in payload.get("artifacts", ()):
            if not isinstance(item, dict):
                continue
            sha256 = item.get("sha256")
            item_path = item.get("path")
            if not isinstance(sha256, str) or not isinstance(item_path, str):
                continue
            bytes_value = item.get("bytes")
            references.append(
                CheckpointArtifactReference(
                    source_path=artifact.path,
                    kind=str(item.get("kind", "artifact")),
                    path=item_path,
                    bytes=bytes_value if isinstance(bytes_value, int) else None,
                    sha256=sha256,
                )
            )
    return tuple(references)


def _artifact_markdown(artifacts: tuple[FileArtifact, ...]) -> str:
    if not artifacts:
        return "- none"
    return "\n".join(
        f"- `{artifact.path}` ({artifact.bytes} bytes, sha256 `{artifact.sha256}`)"
        for artifact in artifacts
    )


def _model_reference_markdown(references: tuple[ModelReference, ...]) -> str:
    if not references:
        return "- none"
    lines: list[str] = []
    for reference in references:
        revision = reference.revision or "unpinned"
        tokenizer = reference.tokenizer or reference.name
        tokenizer_revision = reference.tokenizer_revision or "unpinned"
        lines.append(
            f"- `{reference.source_path}` `{reference.config_path}` "
            f"({reference.role}): model `{reference.name}` @ `{revision}`, "
            f"tokenizer `{tokenizer}` @ `{tokenizer_revision}`"
        )
    return "\n".join(lines)


def _checkpoint_markdown(
    manifests: tuple[FileArtifact, ...],
    artifacts: tuple[CheckpointArtifactReference, ...],
) -> str:
    if not manifests and not artifacts:
        return "- none"
    lines = [
        f"- manifest `{manifest.path}` ({manifest.bytes} bytes, sha256 `{manifest.sha256}`)"
        for manifest in manifests
    ]
    lines.extend(
        f"- `{artifact.kind}` from `{artifact.source_path}`: `{artifact.path}` "
        f"(sha256 `{artifact.sha256}`)"
        for artifact in artifacts
    )
    return "\n".join(lines)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _git_output(root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _git_status(root: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _package_version(root: Path) -> str:
    init_path = root / "src" / "codeself" / "__init__.py"
    if not init_path.exists():
        return "unknown"
    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip("\"'")
    return "unknown"
