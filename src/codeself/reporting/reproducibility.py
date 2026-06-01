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
            purpose="Exercises dataset loading, sandbox execution, rewards, rollouts, training smoke tests, and reports.",
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
                "python3 scripts/evaluate_rollouts.py --rollouts outputs/rollouts/mock_smoke.jsonl "
                "--output outputs/reports/mock_baseline.md --power-output outputs/reports/mock_power.json"
            ),
            purpose="Summarizes pass@k, parser failures, reward statistics, and power-planning diagnostics.",
        ),
        CommandSpec(
            name="Run GRPO smoke diagnostics",
            command=(
                "python3 scripts/train_grpo_smoke.py --tasks configs/datasets/tasks.example.jsonl "
                "--output-dir outputs/checkpoints/grpo_smoke_example --backend mock --group-size 4 --max-steps 3"
            ),
            purpose="Validates grouped reward advantages and checkpoint/report plumbing without model fine-tuning.",
        ),
        CommandSpec(
            name="Run PPO smoke diagnostics",
            command=(
                "python3 scripts/train_ppo_smoke.py --tasks configs/datasets/tasks.example.jsonl "
                "--output-dir outputs/checkpoints/ppo_smoke_example --backend mock --samples-per-task 4 --max-steps 3"
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
            purpose="Records artifact checksums, environment metadata, git status, and rerun commands.",
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
    generated_at = generated_at_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifacts = collect_artifacts(root_path)
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
        commands=default_reproduction_commands(),
        notes=(
            "Smoke commands use dependency-free mock generation and do not fine-tune model weights.",
            "Real training artifacts must add model revision, tokenizer revision, adapter checksum, hardware, and run seeds.",
            "Private hidden tests should be archived separately and never embedded in prompts or public rollouts.",
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
