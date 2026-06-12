"""Static safety checks for generated Python code.

The scan is intentionally conservative about *dangerous* surfaces while
staying permissive about ordinary Python. It is one layer of three:

1. this static scan rejects obvious escape attempts before execution;
2. runtime startup hardening (``sitecustomize``) blocks file/network/exit
   surfaces inside the child process;
3. the subprocess jail enforces rlimits, process-group kill, and a
   completion sentinel so a forced clean exit cannot fake a pass.

Importantly, the scan must not reject idiomatic solution code. Generic
method names such as ``str.replace`` or ``list.remove`` and reflective
builtins such as ``getattr``/``hasattr`` are allowed; only dunder-level
introspection escapes and dangerous module imports are blocked.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from codeself.execution.results import SecurityFinding


BLOCKED_MODULES = {
    "_io",
    "_posixsubprocess",
    "_socket",
    "_thread",
    "asyncio",
    "builtins",
    "codecs",
    "ctypes",
    "faulthandler",
    "fcntl",
    "ftplib",
    "gc",
    "glob",
    "http",
    "importlib",
    "inspect",
    "io",
    "marshal",
    "mmap",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "pty",
    "requests",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "traceback",
    "urllib",
    "webbrowser",
}

BLOCKED_CALLS = {
    "SystemExit",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "input",
    "locals",
    "open",
    "quit",
    "vars",
}

# Reflective builtins are allowed in general, but not when used to reach a
# blocked dunder attribute through a string literal.
REFLECTIVE_CALLS = {"delattr", "getattr", "hasattr", "setattr"}

BLOCKED_ATTRIBUTES = {
    "__base__",
    "__bases__",
    "__builtins__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__import__",
    "__loader__",
    "__mro__",
    "__reduce__",
    "__reduce_ex__",
    "__self__",
    "__spec__",
    "__subclasses__",
    "ag_frame",
    "cr_frame",
    "f_back",
    "f_builtins",
    "f_globals",
    "f_locals",
    "gi_frame",
    "mro",
    "tb_frame",
}

BLOCKED_RAISES = {
    "GeneratorExit",
    "KeyboardInterrupt",
    "SystemExit",
}


@dataclass(frozen=True)
class SecurityScanResult:
    """Static scan result."""

    allowed: bool
    findings: tuple[SecurityFinding, ...]


def scan_python_code(code: str) -> SecurityScanResult:
    """Scan generated Python code for obviously unsafe behavior."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        finding = SecurityFinding(
            rule="syntax",
            message=exc.msg,
            line=exc.lineno,
            severity="error",
        )
        return SecurityScanResult(allowed=False, findings=(finding,))

    visitor = _SecurityVisitor()
    visitor.visit(tree)
    return SecurityScanResult(allowed=not visitor.findings, findings=tuple(visitor.findings))


class _SecurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[SecurityFinding] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            module = alias.name
            if _module_blocked(module):
                self._add("blocked_import", f"blocked import: {module}", node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if _module_blocked(module):
            self._add("blocked_import", f"blocked import: {module}", node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        if name in BLOCKED_CALLS:
            self._add("blocked_call", f"blocked call: {name}", node.lineno)
        if name in REFLECTIVE_CALLS:
            for argument in node.args:
                literal = _string_literal(argument)
                if literal in BLOCKED_ATTRIBUTES:
                    self._add(
                        "blocked_reflection",
                        f"blocked reflective access: {name}(..., {literal!r})",
                        node.lineno,
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr in BLOCKED_ATTRIBUTES:
            self._add("blocked_attribute", f"blocked attribute access: {node.attr}", node.lineno)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
        raised = _raised_name(node.exc)
        if raised in BLOCKED_RAISES:
            self._add("blocked_raise", f"blocked raise: {raised}", node.lineno)
        self.generic_visit(node)

    def _add(self, rule: str, message: str, line: int | None) -> None:
        self.findings.append(SecurityFinding(rule=rule, message=message, line=line))


def _module_blocked(module: str) -> bool:
    root = module.split(".", 1)[0]
    return module in BLOCKED_MODULES or root in BLOCKED_MODULES


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _raised_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _call_name(node.func)
    return None
