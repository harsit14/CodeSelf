"""Dependency-free subprocess runner for training rollouts."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from codeself.datasets import ResourceLimits
from codeself.execution.harness import write_phase_files
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
            write_phase_files(temp_path, candidate_code=candidate_code, phase_code=phase_code)

            start = time.monotonic()
            process = subprocess.Popen(
                [self.config.python_executable, "run_phase.py"],
                cwd=temp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_clean_environment(temp_path),
                preexec_fn=_resource_limiter(limits),
                start_new_session=platform.system() != "Windows",
            )
            try:
                stdout, stderr = process.communicate(timeout=limits.timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                stdout, stderr = process.communicate()
                duration = time.monotonic() - start
                return PhaseResult(
                    name=phase_name,
                    status=PhaseStatus.TIMEOUT,
                    duration_seconds=duration,
                    stdout=_truncate(stdout or "", self.config.max_output_chars),
                    stderr=_truncate(stderr or "", self.config.max_output_chars),
                    error=f"timed out after {limits.timeout_seconds:.3f}s",
                    tests_run=tests_run,
                )

            duration = time.monotonic() - start
            stdout = _truncate(stdout, self.config.max_output_chars)
            stderr = _truncate(stderr, self.config.max_output_chars)
            status = PhaseStatus.PASSED if process.returncode == 0 else PhaseStatus.FAILED
            return PhaseResult(
                name=phase_name,
                status=status,
                duration_seconds=duration,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                error="" if status == PhaseStatus.PASSED else _last_line(stderr),
                tests_run=tests_run,
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

        if hasattr(resource, "RLIMIT_NOFILE"):
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
            except (ValueError, OSError):
                pass

        if hasattr(resource, "RLIMIT_FSIZE"):
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
            except (ValueError, OSError):
                pass

        if hasattr(resource, "RLIMIT_CORE"):
            try:
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except (ValueError, OSError):
                pass

    return limit_resources


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if platform.system() != "Windows":
        try:
            os.killpg(process.pid, 9)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    try:
        process.kill()
    except OSError:
        pass


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
