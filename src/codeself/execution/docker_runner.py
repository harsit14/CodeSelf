"""Docker runner for stricter final-evaluation isolation."""

from __future__ import annotations

import secrets
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codeself.datasets import ResourceLimits
from codeself.execution.harness import phase_sentinel, phase_stdin_payload, write_phase_files
from codeself.execution.results import PhaseResult, PhaseStatus


CommandRunner = Callable[[list[str], float, str], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Docker image/runtime settings for final evaluation."""

    docker_executable: str = "docker"
    image: str = "codeself-executor:latest"
    cpus: float = 1.0
    pids_limit: int = 64
    user: str = "1000:1000"
    workdir: str = "/workspace"
    tmpfs_size: str = "16m"
    max_output_chars: int = 12_000


class DockerSandboxRunner:
    """Runs Python snippets in a locked-down Docker container."""

    def __init__(
        self,
        config: DockerSandboxConfig | None = None,
        *,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config or DockerSandboxConfig()
        self._command_runner = command_runner or _default_command_runner

    def build_command(self, task_dir: str | Path, limits: ResourceLimits) -> list[str]:
        task_path = Path(task_dir).resolve()
        return [
            self.config.docker_executable,
            "run",
            "--rm",
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config.pids_limit),
            "--cpus",
            str(self.config.cpus),
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--user",
            self.config.user,
            "--workdir",
            self.config.workdir,
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={self.config.tmpfs_size}",
            "--mount",
            f"type=bind,source={task_path},target={self.config.workdir},readonly",
            self.config.image,
            "python",
            "run_phase.py",
        ]

    def run_phase(
        self,
        *,
        phase_name: str,
        candidate_code: str,
        phase_code: str,
        limits: ResourceLimits,
        tests_run: int,
    ) -> PhaseResult:
        """Run one phase through Docker and return a standard phase result.

        Docker itself is not required for unit tests; callers can inject a
        command runner that returns a `subprocess.CompletedProcess`.
        """

        with tempfile.TemporaryDirectory(prefix="codeself-docker-run-") as tmpdir:
            temp_path = Path(tmpdir)
            write_phase_files(temp_path, candidate_code=candidate_code)
            nonce = secrets.token_hex(16)
            sentinel = phase_sentinel(nonce)
            command = self.build_command(temp_path, limits)
            start = time.monotonic()
            try:
                completed = self._command_runner(
                    command,
                    limits.timeout_seconds,
                    phase_stdin_payload(nonce, phase_code),
                )
            except subprocess.TimeoutExpired as exc:
                duration = time.monotonic() - start
                return PhaseResult(
                    name=phase_name,
                    status=PhaseStatus.TIMEOUT,
                    duration_seconds=duration,
                    stdout=_truncate(exc.stdout or "", self.config.max_output_chars),
                    stderr=_truncate(exc.stderr or "", self.config.max_output_chars),
                    error=f"docker phase timed out after {limits.timeout_seconds:.3f}s",
                    tests_run=tests_run,
                )
            except FileNotFoundError as exc:
                duration = time.monotonic() - start
                return PhaseResult(
                    name=phase_name,
                    status=PhaseStatus.RUNTIME_ERROR,
                    duration_seconds=duration,
                    error=f"docker executable not found: {self.config.docker_executable}",
                    tests_run=tests_run,
                )

            duration = time.monotonic() - start
            sentinel_present = sentinel in (completed.stdout or "")
            stdout = _truncate(
                (completed.stdout or "").replace(sentinel, ""), self.config.max_output_chars
            )
            stderr = _truncate(completed.stderr, self.config.max_output_chars)
            if completed.returncode == 0 and sentinel_present:
                status = PhaseStatus.PASSED
                error = ""
            elif completed.returncode == 0:
                status = PhaseStatus.RUNTIME_ERROR
                error = "phase exited cleanly without completing the harness (possible forced exit)"
            else:
                status = PhaseStatus.FAILED
                error = _last_line(stderr)
            return PhaseResult(
                name=phase_name,
                status=status,
                duration_seconds=duration,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                error=error,
                tests_run=tests_run,
            )


def _default_command_runner(
    command: list[str],
    timeout_seconds: float,
    stdin_payload: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=stdin_payload,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


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
