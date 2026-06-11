"""Shared Python phase harness helpers."""

from __future__ import annotations

import textwrap
from pathlib import Path


def write_phase_files(temp_path: Path, *, candidate_code: str, phase_code: str) -> None:
    """Write the files needed to run one candidate/test phase."""

    (temp_path / "sitecustomize.py").write_text(sitecustomize_code(), encoding="utf-8")
    (temp_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
    (temp_path / "run_phase.py").write_text(runner_code(phase_code), encoding="utf-8")


def runner_code(phase_code: str) -> str:
    """Return the Python test harness code for one phase."""

    indented = textwrap.indent(phase_code, "    ")
    return (
        "from candidate import *\n\n"
        "def __codeself_run_phase():\n"
        f"{indented if indented.strip() else '    pass'}\n\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        __codeself_run_phase()\n"
        "    except (SystemExit, KeyboardInterrupt, GeneratorExit) as exc:\n"
        "        raise RuntimeError("
        "\"blocked control-flow exception: \" + type(exc).__name__"
        ") from exc\n"
    )


def sitecustomize_code() -> str:
    """Return startup hardening loaded before candidate code imports."""

    return """
import builtins
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


socket.socket = _codeself_blocked_network
socket.create_connection = _codeself_blocked_network
"""
