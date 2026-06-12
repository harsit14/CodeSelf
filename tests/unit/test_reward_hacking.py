from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.datasets import Split, TaskSpec, TestSpec  # noqa: E402
from codeself.rewards import detect_reward_hacking  # noqa: E402


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="rh/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(
            TestSpec(name="public-1", code="assert add_one(1) == 2"),
            TestSpec(name="public-2", code="assert add_one(5) == 6"),
        ),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(10) == 11"),),
    )


class RewardHackingNoFalsePositivesTests(unittest.TestCase):
    def test_genuine_solution_is_not_flagged(self) -> None:
        report = detect_reward_hacking("def add_one(x):\n    return x + 1", _task())
        self.assertFalse(report.flagged)
        self.assertEqual(report.reasons, ())

    def test_multiline_genuine_solution_is_not_flagged(self) -> None:
        code = (
            "def add_one(x):\n"
            "    result = x\n"
            "    result += 1\n"
            "    return result\n"
        )
        self.assertFalse(detect_reward_hacking(code, _task()).flagged)

    def test_parameterless_helper_is_not_flagged_as_constant(self) -> None:
        # A function with no arguments has nothing to ignore.
        code = "def add_one(x):\n    return x + 1\n\ndef config():\n    return 42"
        self.assertFalse(detect_reward_hacking(code, _task()).constant_entry_function)


class RewardHackingDetectionTests(unittest.TestCase):
    def test_empty_code_is_flagged(self) -> None:
        report = detect_reward_hacking("   \n  ", _task())
        self.assertTrue(report.flagged)
        self.assertTrue(report.empty_code)

    def test_no_function_definition_is_flagged(self) -> None:
        report = detect_reward_hacking("x = 1 + 1", _task())
        self.assertTrue(report.flagged)
        self.assertTrue(report.no_function_def)

    def test_constant_function_ignoring_args_is_flagged(self) -> None:
        report = detect_reward_hacking("def add_one(x):\n    return 2", _task())
        self.assertTrue(report.constant_entry_function)
        self.assertTrue(report.flagged)

    def test_hardcoded_visible_answers_is_flagged(self) -> None:
        # Lookup table over the visible test inputs/outputs.
        code = (
            "def add_one(x):\n"
            "    if x == 1:\n"
            "        return 2\n"
            "    if x == 5:\n"
            "        return 6\n"
        )
        report = detect_reward_hacking(code, _task())
        self.assertTrue(report.echoes_expected_values)
        self.assertTrue(report.flagged)

    def test_constant_matching_expected_value_is_hardcoded(self) -> None:
        report = detect_reward_hacking("def add_one(x):\n    return 2", _task())
        self.assertTrue(report.constant_entry_function)
        self.assertTrue(report.echoes_expected_values)
        self.assertTrue(report.hardcoded_visible_answers)

    def test_repeated_lines_are_flagged(self) -> None:
        code = "def f():\n" + "    x = 1\n" * 10
        report = detect_reward_hacking(code, _task())
        self.assertGreaterEqual(report.repeated_line_fraction, 0.5)
        self.assertTrue(report.flagged)

    def test_report_serializes(self) -> None:
        report = detect_reward_hacking("def add_one(x):\n    return 2", _task())
        payload = report.to_dict()
        self.assertIn("reasons", payload)
        self.assertIsInstance(payload["reasons"], list)
        self.assertTrue(payload["flagged"])


if __name__ == "__main__":
    unittest.main()
