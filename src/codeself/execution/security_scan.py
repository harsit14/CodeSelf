"""Static safety checks for generated Python code.

The scan is intentionally conservative. It is not a substitute for runtime
isolation, but it rejects obvious file, process, and network behavior before
the executor spends resources on a candidate.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from codeself.execution.results import SecurityFinding


BLOCKED_MODULES = {
    "asyncio.subprocess",
    "ctypes",
    "ftplib",
    "glob",
    "http",
    "http.client",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
    "urllib.request",
}

BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "delattr",
    "getattr",
    "globals",
    "hasattr",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "SystemExit",
    "vars",
}

BLOCKED_ATTRIBUTES = {
    "__builtins__",
    "__class__",
    "__code__",
    "__closure__",
    "__dict__",
    "__getattribute__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "connect",
    "kill",
    "mro",
    "mkdir",
    "open",
    "popen",
    "read",
    "read_text",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "socket",
    "spawn",
    "system",
    "unlink",
    "write",
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
        attribute = _attribute_name(node.func)
        if attribute in BLOCKED_ATTRIBUTES:
            self._add("blocked_attribute", f"blocked attribute call: {attribute}", node.lineno)
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
