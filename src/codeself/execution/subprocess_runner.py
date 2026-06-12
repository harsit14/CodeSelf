"""Dependency-free subprocess runner for training rollouts."""

from __future__ import annotations

import math
import os
import platform
import secrets
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from codeself.datasets import ResourceLimits
from codeself.execution.harness import phase_sentinel, phase_stdin_payload, write_phase_files
from codeself.execution.results import PhaseResult, PhaseStatus


@dataclass(frozen=True)
class SubprocessSandboxConfig:
    """Configuration for the local subprocess sandbox."""

    python_executable: str = sys.executable
    max_output_chars: int = 12_000
    memory_watchdog_interval_seconds: float = 0.1


class SubprocessSandboxRunner:
    """Runs Python snippets in a temp directory with time/resource limits.

    Phase/test code is sent to the child over stdin and never written to
    disk. A phase counts as passed only when the child exits cleanly *and*
    emits a per-run nonce sentinel, so forced clean exits cannot fake a pass.
    """

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
            write_phase_files(temp_path, candidate_code=candidate_code)
            nonce = secrets.token_hex(16)
            sentinel = phase_sentinel(nonce)

            start = time.monotonic()
            process = subprocess.Popen(
                [self.config.python_executable, "run_phase.py"],
                cwd=temp_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=_clean_environment(temp_path),
                preexec_fn=_resource_limiter(limits),
                start_new_session=platform.system() != "Windows",
            )
            watchdog = _MemoryWatchdog(
                process,
                memory_bytes=limits.memory_mb * 1024 * 1024,
                interval_seconds=self.config.memory_watchdog_interval_seconds,
            )
            watchdog.start()
            try:
                stdout, stderr = process.communicate(
                    input=phase_stdin_payload(nonce, phase_code),
                    timeout=limits.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                stdout, stderr = process.communicate()
                watchdog.stop()
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
            finally:
                watchdog.stop()

            duration = time.monotonic() - start
            if watchdog.triggered:
                return PhaseResult(
                    name=phase_name,
                    status=PhaseStatus.RUNTIME_ERROR,
                    duration_seconds=duration,
                    exit_code=process.returncode,
                    stdout=_truncate(_strip_sentinel(stdout, sentinel), self.config.max_output_chars),
                    stderr=_truncate(stderr, self.config.max_output_chars),
                    error=f"memory limit exceeded ({limits.memory_mb} MB)",
                    tests_run=tests_run,
                )

            sentinel_present = sentinel in (stdout or "")
            stdout = _truncate(_strip_sentinel(stdout, sentinel), self.config.max_output_chars)
            stderr = _truncate(stderr, self.config.max_output_chars)
            if process.returncode == 0 and sentinel_present:
                status = PhaseStatus.PASSED
                error = ""
            elif process.returncode == 0:
                status = PhaseStatus.RUNTIME_ERROR
                error = "phase exited cleanly without completing the harness (possible forced exit)"
            else:
                status = PhaseStatus.FAILED
                error = _last_line(stderr)
            return PhaseResult(
                name=phase_name,
                status=status,
                duration_seconds=duration,
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                error=error,
                tests_run=tests_run,
            )


class _MemoryWatchdog:
    """Parent-side RSS watchdog for hosts without enforceable memory rlimits.

    macOS refuses ``setrlimit(RLIMIT_DATA/AS)`` reductions, so the parent
    polls the child's resident set size and kills the process group when it
    exceeds the configured budget.
    """

    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        memory_bytes: int,
        interval_seconds: float,
    ) -> None:
        self._process = process
        self._memory_bytes = memory_bytes
        self._interval = max(0.02, interval_seconds)
        self._stop_event = threading.Event()
        self.triggered = False
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def start(self) -> None:
        if self._memory_bytes > 0 and platform.system() != "Windows":
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _watch(self) -> None:
        while not self._stop_event.wait(self._interval):
            if self._process.poll() is not None:
                return
            rss = _read_rss_bytes(self._process.pid)
            if rss is not None and rss > self._memory_bytes:
                self.triggered = True
                _terminate_process_tree(self._process)
                return


def _read_rss_bytes(pid: int) -> int | None:
    status_path = Path(f"/proc/{pid}/status")
    if status_path.exists():
        try:
            for line in status_path.read_text(encoding="ascii", errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        text = completed.stdout.strip()
        return int(text) * 1024 if text else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def _strip_sentinel(stdout: str | None, sentinel: str) -> str:
    return (stdout or "").replace(sentinel, "")


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

        memory_bytes = limits.memory_mb * 1024 * 1024
        if platform.system() != "Darwin" and hasattr(resource, "RLIMIT_AS"):
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            except (ValueError, OSError):
                pass
        if hasattr(resource, "RLIMIT_DATA"):
            # Enforced on Linux; macOS rejects the call, which is why the
            # parent-side memory watchdog exists.
            try:
                resource.setrlimit(resource.RLIMIT_DATA, (memory_bytes, memory_bytes))
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
