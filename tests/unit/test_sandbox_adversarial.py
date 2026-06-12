"""Adversarial sandbox tests.

Each test encodes one escape/abuse vector and proves it is contained by at
least one sandbox layer (static scan, runtime hardening, subprocess jail,
or the completion-sentinel protocol).
"""

from __future__ import annotations

import platform
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import ResourceLimits  # noqa: E402
from codeself.execution import PhaseStatus, SubprocessSandboxRunner  # noqa: E402
from codeself.execution.harness import write_phase_files  # noqa: E402
from codeself.execution.security_scan import scan_python_code  # noqa: E402


LIMITS = ResourceLimits(timeout_seconds=5.0, memory_mb=256)


def _run_phase(
    candidate_code: str,
    *,
    phase_code: str = "assert True",
    limits: ResourceLimits = LIMITS,
):
    return SubprocessSandboxRunner().run_phase(
        phase_name="public_tests",
        candidate_code=candidate_code,
        phase_code=phase_code,
        limits=limits,
        tests_run=1,
    )


class StaticScanAllowsIdiomaticPythonTests(unittest.TestCase):
    """The scanner must not punish ordinary solution code."""

    def test_common_method_names_are_allowed(self) -> None:
        snippets = (
            'def f(s):\n    return s.replace("a", "b")',
            "def f(xs):\n    xs.remove(1)\n    return xs",
            "def f(xs):\n    xs.append(2)\n    return sorted(xs)",
            'def f(d):\n    return d.read if hasattr(d, "read") else None',
            'def f(o):\n    return getattr(o, "value", 0)',
            'def f(o):\n    setattr(o, "value", 1)\n    return o',
            "def f(a, b):\n    return a.union(b)",
            'def f(text):\n    return text.split(",")',
        )
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                self.assertTrue(scan_python_code(snippet).allowed)

    def test_idiomatic_solution_passes_end_to_end(self) -> None:
        result = _run_phase(
            "def dedupe(xs):\n"
            "    out = list(dict.fromkeys(xs))\n"
            "    if 0 in out:\n"
            "        out.remove(0)\n"
            "    return out\n",
            phase_code="assert dedupe([1, 1, 0, 2]) == [1, 2]",
        )
        self.assertEqual(result.status, PhaseStatus.PASSED)


class StaticScanBlocksEscapeSurfacesTests(unittest.TestCase):
    def test_dangerous_imports_are_rejected(self) -> None:
        for module in ("os", "sys", "io", "_io", "gc", "inspect", "codecs",
                       "builtins", "socket", "subprocess", "ctypes", "importlib"):
            with self.subTest(module=module):
                result = scan_python_code(f"import {module}")
                self.assertFalse(result.allowed)

    def test_file_io_class_is_rejected_via_import(self) -> None:
        result = scan_python_code('import io\n\ndef f():\n    return io.FileIO("run_phase.py")')
        self.assertFalse(result.allowed)

    def test_reflective_access_to_blocked_dunder_is_rejected(self) -> None:
        result = scan_python_code('def f(x):\n    return getattr(x, "__globals__")')
        self.assertFalse(result.allowed)
        self.assertTrue(any(f.rule == "blocked_reflection" for f in result.findings))

    def test_introspection_dunders_are_rejected(self) -> None:
        for snippet in (
            "def f(x):\n    return x.__class__",
            "def f(x):\n    return type(x).__mro__",
            "def f():\n    return ().__class__.__bases__[0].__subclasses__()",
            "def f(g):\n    return g.gi_frame.f_globals",
        ):
            with self.subTest(snippet=snippet):
                self.assertFalse(scan_python_code(snippet).allowed)

    def test_exec_eval_compile_and_import_calls_are_rejected(self) -> None:
        for snippet in (
            'def f():\n    return eval("1+1")',
            'def f():\n    exec("x = 1")',
            'def f():\n    return compile("1", "<s>", "eval")',
            'def f():\n    return __import__("os")',
            'def f():\n    return open("run_phase.py")',
        ):
            with self.subTest(snippet=snippet):
                self.assertFalse(scan_python_code(snippet).allowed)

    def test_control_flow_exit_raises_are_rejected(self) -> None:
        for snippet in (
            "def f():\n    raise SystemExit(0)",
            "def f():\n    raise KeyboardInterrupt",
            "def f():\n    exit()",
        ):
            with self.subTest(snippet=snippet):
                self.assertFalse(scan_python_code(snippet).allowed)


class HarnessLeakTests(unittest.TestCase):
    def test_phase_code_is_never_written_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            write_phase_files(temp_path, candidate_code="def f():\n    return 1")
            files = sorted(p.name for p in temp_path.iterdir())
            self.assertEqual(files, ["candidate.py", "run_phase.py", "sitecustomize.py"])
            harness_text = (temp_path / "run_phase.py").read_text(encoding="utf-8")
            self.assertIn("sys.stdin.read()", harness_text)

    def test_candidate_reading_harness_learns_no_test_content(self) -> None:
        # Even with raw os-level file access (bypassing the static scan and
        # the patched open/io surfaces), the harness file contains only
        # generic runner code -- the asserts travel via stdin.
        candidate = (
            "import os\n"
            "fd = os.open('run_phase.py', os.O_RDONLY)\n"
            "content = b''\n"
            "while True:\n"
            "    chunk = os.read(fd, 65536)\n"
            "    if not chunk:\n"
            "        break\n"
            "    content += chunk\n"
            "os.close(fd)\n"
            "assert b'SECRET_EXPECTED_VALUE' not in content\n"
            "def f():\n"
            "    return 41\n"
        )
        result = _run_phase(candidate, phase_code="assert f() == 42, 'SECRET_EXPECTED_VALUE'")
        # The candidate found no secret on disk (its own assert passed) and
        # still failed the real test delivered over stdin.
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("SECRET_EXPECTED_VALUE", result.stderr)

    def test_candidate_cannot_reread_stdin_for_phase_source(self) -> None:
        candidate = (
            "import sys\n"
            "leftover = sys.stdin.read()\n"
            "assert 'SECRET' not in leftover\n"
            "def f():\n"
            "    return 0\n"
        )
        result = _run_phase(candidate, phase_code="assert f() == 1, 'SECRET'")
        self.assertEqual(result.status, PhaseStatus.FAILED)


class ForcedExitTests(unittest.TestCase):
    def test_clean_exit_without_sentinel_is_not_a_pass(self) -> None:
        result = _run_phase("import os\nos._exit(0)\n")
        self.assertEqual(result.status, PhaseStatus.RUNTIME_ERROR)
        self.assertIn("forced exit", result.error)

    def test_forged_sentinel_with_wrong_nonce_is_not_a_pass(self) -> None:
        candidate = (
            "import os\n"
            "print('<<CODESELF_PHASE_OK:' + '0' * 32 + '>>', end='')\n"
            "os._exit(0)\n"
        )
        result = _run_phase(candidate)
        self.assertEqual(result.status, PhaseStatus.RUNTIME_ERROR)

    def test_import_time_system_exit_is_converted_to_failure(self) -> None:
        result = _run_phase("raise SystemExit\n")
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("blocked control-flow exception", result.stderr)

    def test_test_time_system_exit_is_converted_to_failure(self) -> None:
        result = _run_phase(
            "def f():\n    raise SystemExit(0)\n",
            phase_code="f()",
        )
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("blocked control-flow exception", result.stderr)


class RuntimeHardeningTests(unittest.TestCase):
    def test_io_fileio_is_blocked_at_runtime(self) -> None:
        result = _run_phase(
            "import io\n"
            "def f():\n"
            "    return io.FileIO('run_phase.py').readall()\n",
            phase_code="f()",
        )
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("disabled by CodeSelf sandbox", result.stderr)

    def test_codecs_open_is_blocked_at_runtime(self) -> None:
        result = _run_phase(
            "import codecs\n"
            "def f():\n"
            "    return codecs.open('run_phase.py').read()\n",
            phase_code="f()",
        )
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("disabled by CodeSelf sandbox", result.stderr)

    def test_network_socket_is_blocked_at_runtime(self) -> None:
        result = _run_phase(
            "import socket\n"
            "def f():\n"
            "    return socket.socket()\n",
            phase_code="f()",
        )
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("network access is disabled", result.stderr)

    def test_eval_is_blocked_at_runtime(self) -> None:
        result = _run_phase(
            "def f():\n    return eval('1 + 1')\n",
            phase_code="f()",
        )
        self.assertEqual(result.status, PhaseStatus.FAILED)
        self.assertIn("disabled by CodeSelf sandbox", result.stderr)


@unittest.skipIf(platform.system() == "Windows", "POSIX resource limits required")
class ResourceAbuseTests(unittest.TestCase):
    def test_fork_bomb_is_contained(self) -> None:
        start = time.monotonic()
        result = _run_phase(
            "import os\n"
            "while True:\n"
            "    os.fork()\n",
            limits=ResourceLimits(timeout_seconds=2.0, memory_mb=256),
        )
        elapsed = time.monotonic() - start
        self.assertNotEqual(result.status, PhaseStatus.PASSED)
        self.assertLess(elapsed, 30.0)

    def test_memory_bomb_is_contained(self) -> None:
        # RLIMIT_AS/DATA stops this on Linux; the parent-side RSS watchdog
        # stops it on macOS where those rlimits are not enforceable.
        result = _run_phase(
            "import time\n"
            "x = b'x' * (1024 ** 3)\n"
            "time.sleep(8)\n"
            "def f():\n"
            "    return 1\n",
            limits=ResourceLimits(timeout_seconds=10.0, memory_mb=256),
        )
        self.assertNotEqual(result.status, PhaseStatus.PASSED)
        self.assertIn(
            result.status,
            (PhaseStatus.RUNTIME_ERROR, PhaseStatus.FAILED, PhaseStatus.TIMEOUT),
        )

    def test_infinite_loop_times_out(self) -> None:
        start = time.monotonic()
        result = _run_phase(
            "def f():\n    return 1\n",
            phase_code="while True:\n    pass",
            limits=ResourceLimits(timeout_seconds=0.5, memory_mb=256),
        )
        elapsed = time.monotonic() - start
        self.assertEqual(result.status, PhaseStatus.TIMEOUT)
        self.assertLess(elapsed, 10.0)


if __name__ == "__main__":
    unittest.main()
