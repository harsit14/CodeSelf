"""Dependency-free subprocess runner for training rollouts."""

from __future__ import annotations

import math
import platform
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from codeself.datasets import ResourceLimits
from codeself.execution.results import PhaseResult, PhaseStatus


@dataclass(frozen=True)
class SubprocessSandboxConfig:
    """Configuration for the local subprocess sandbox."""

    python_executable: str = sys.executable
    max_output_chars: int = 12_000


class SubprocessSandboxRunner:
    """Runs Python snippets in a temp directory with time/resource limits."""

    def __init__(self, config: SubprocessSandboxConfig | None = None) -> None:
        self.config = config or SubprocessSandboxConfig()

    def run_phase(
        self,
        *,
        phase_name: str,
        candidate_code: str,
        phase_code: str,
        limits: ResourceLimits,
        tests_run: int,
    ) -> PhaseResult:
        with tempfile.TemporaryDirectory(prefix="codeself-run-") as tmpdir:
            temp_path = Path(tmpdir)
            (temp_path / "sitecustomize.py").write_text(_sitecustomize_code(), encoding="utf-8")
            (temp_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
            (temp_path / "run_phase.py").write_text(_runner_code(phase_code), encoding="utf-8")

            start = time.monotonic()
            try:
                completed = subprocess.run(
                    [self.config.python_executable, "run_phase.py"],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=limits.timeout_seconds,
                    env=_clean_environment(temp_path),
                    preexec_fn=_resource_limiter(limits),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - start
                return PhaseResult(
                    name=phase_name,
                    status=PhaseStatus.TIMEOUT,
                    duration_seconds=duration,
                    stdout=_truncate(exc.stdout or "", self.config.max_output_chars),
                    stderr=_truncate(exc.stderr or "", self.config.max_output_chars),
                    error=f"timed out after {limits.timeout_seconds:.3f}s",
                    tests_run=tests_run,
                )

            duration = time.monotonic() - start
            stdout = _truncate(completed.stdout, self.config.max_output_chars)
            stderr = _truncate(completed.stderr, self.config.max_output_chars)
            status = PhaseStatus.PASSED if completed.returncode == 0 else PhaseStatus.FAILED
            return PhaseResult(
                name=phase_name,
                status=status,
                duration_seconds=duration,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                error="" if status == PhaseStatus.PASSED else _last_line(stderr),
                tests_run=tests_run,
            )


def _runner_code(phase_code: str) -> str:
    indented = textwrap.indent(phase_code, "    ")
    return (
        "from candidate import *\n\n"
        "def __codeself_run_phase():\n"
        f"{indented if indented.strip() else '    pass'}\n\n"
        "if __name__ == '__main__':\n"
        "    __codeself_run_phase()\n"
    )


def _clean_environment(temp_path: Path) -> dict[str, str]:
    return {
        "PYTHONPATH": str(temp_path),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C",
    }


def _resource_limiter(limits: ResourceLimits):
    if platform.system() == "Windows":
        return None

    def limit_resources() -> None:
        import resource

        cpu_seconds = max(1, int(math.ceil(limits.timeout_seconds)) + 1)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass

        if platform.system() != "Darwin" and hasattr(resource, "RLIMIT_AS"):
            memory_bytes = limits.memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            except (ValueError, OSError):
                pass

        if hasattr(resource, "RLIMIT_NPROC"):
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))
            except (ValueError, OSError):
                pass

    return limit_resources


def _sitecustomize_code() -> str:
    return """
import socket


def _codeself_blocked_network(*args, **kwargs):
    raise RuntimeError("network access is disabled by CodeSelf sandbox")


socket.socket = _codeself_blocked_network
socket.create_connection = _codeself_blocked_network
"""


def _truncate(value: str | bytes, limit: int) -> str:
    text = value.decode(errors="replace") if isinstance(value, bytes) else value
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...<truncated>..."


def _last_line(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[-1]
