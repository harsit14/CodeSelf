"""Shared Python phase harness helpers.

Security design notes:

- Test/phase code is delivered to the child over **stdin**, never written to
  disk, so candidate code cannot read expected outputs out of the run
  directory even if it gains file access.
- stdin is fully consumed and the raw source deleted *before* the candidate
  module is imported, so import-time candidate code cannot re-read it.
- The runner appends a per-run nonce sentinel to stdout on successful
  completion. The parent treats a phase as passed only when the exit code is
  zero *and* the sentinel is present, so ``os._exit(0)``-style forced clean
  exits cannot fake a pass. The nonce arrives on the first stdin line and is
  unknown to the candidate.
- ``SystemExit``/``KeyboardInterrupt``/``GeneratorExit`` raised anywhere in
  candidate import or test execution are converted into ordinary failures.
"""

from __future__ import annotations

from pathlib import Path

SENTINEL_PREFIX = "<<CODESELF_PHASE_OK:"
SENTINEL_SUFFIX = ">>"


def phase_sentinel(nonce: str) -> str:
    """Return the completion sentinel for one run nonce."""

    return f"{SENTINEL_PREFIX}{nonce}{SENTINEL_SUFFIX}"


def phase_stdin_payload(nonce: str, phase_code: str) -> str:
    """Return the stdin payload carrying the nonce and phase source."""

    return f"{nonce}\n{phase_code}"


def write_phase_files(temp_path: Path, *, candidate_code: str) -> None:
    """Write the files needed to run one candidate/test phase.

    Only the candidate code and generic harness code are materialized;
    phase/test code is passed via stdin at run time.
    """

    (temp_path / "sitecustomize.py").write_text(sitecustomize_code(), encoding="utf-8")
    (temp_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
    (temp_path / "run_phase.py").write_text(runner_code(), encoding="utf-8")


def runner_code() -> str:
    """Return the generic Python test harness code.

    The harness contains no task-specific content. Protocol: line one of
    stdin is the run nonce, the remainder is the phase source code.
    """

    return f'''
import sys

__codeself_stdin = sys.stdin.read()
__codeself_nonce, __codeself_sep, __codeself_source = __codeself_stdin.partition("\\n")
if not __codeself_sep:
    raise RuntimeError("harness protocol error: missing nonce line")
__codeself_code = compile(__codeself_source, "<codeself-phase>", "exec")
del __codeself_stdin
del __codeself_source


def __codeself_main():
    import candidate

    if hasattr(candidate, "__all__"):
        names = list(candidate.__all__)
    else:
        names = [name for name in dir(candidate) if not name.startswith("_")]
    namespace = {{name: getattr(candidate, name) for name in names}}
    exec(__codeself_code, namespace)


if __name__ == "__main__":
    try:
        __codeself_main()
    except (SystemExit, KeyboardInterrupt, GeneratorExit) as exc:
        raise RuntimeError(
            "blocked control-flow exception: " + type(exc).__name__
        ) from exc
    sys.stdout.flush()
    sys.stdout.write("{SENTINEL_PREFIX}" + __codeself_nonce + "{SENTINEL_SUFFIX}")
    sys.stdout.flush()
'''


def sitecustomize_code() -> str:
    """Return startup hardening loaded before candidate code imports."""

    return """
import builtins
import codecs
import io
import socket


def _codeself_blocked_runtime(*args, **kwargs):
    raise RuntimeError("operation is disabled by CodeSelf sandbox")


def _codeself_blocked_network(*args, **kwargs):
    raise RuntimeError("network access is disabled by CodeSelf sandbox")


for _codeself_name in (
    "breakpoint",
    "eval",
    "exit",
    "input",
    "open",
    "quit",
):
    if hasattr(builtins, _codeself_name):
        setattr(builtins, _codeself_name, _codeself_blocked_runtime)


io.open = _codeself_blocked_runtime
io.open_code = _codeself_blocked_runtime
io.FileIO = _codeself_blocked_runtime
codecs.open = _codeself_blocked_runtime

socket.socket = _codeself_blocked_network
socket.socketpair = _codeself_blocked_network
socket.create_connection = _codeself_blocked_network
socket.create_server = _codeself_blocked_network
socket.fromfd = _codeself_blocked_network
socket.getaddrinfo = _codeself_blocked_network
"""
