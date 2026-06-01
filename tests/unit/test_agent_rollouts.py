from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    DIRECT_SOLUTION_TEMPLATE,
    MockGenerator,
    StaticGenerator,
    extract_code,
    generate_rollouts,
    read_rollouts_jsonl,
    write_rollouts_jsonl,
)
from codeself.datasets import Split, TaskRegistry, TaskSpec, TestSpec  # noqa: E402


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="agent/add-one",
        source="unit-test",
        prompt="Write add_one(x).",
        split=Split.TRAIN,
        entry_point="add_one",
        public_tests=(TestSpec(name="public", code="assert add_one(1) == 2"),),
        hidden_tests=(TestSpec(name="hidden", code="assert add_one(0) == 1"),),
    )


class AgentRolloutTests(unittest.TestCase):
    def test_prompt_template_does_not_include_hidden_tests(self) -> None:
        prompt = DIRECT_SOLUTION_TEMPLATE.render(_task())

        self.assertIn("Write add_one", prompt)
        self.assertNotIn("assert add_one(0) == 1", prompt)

    def test_parser_prefers_python_fenced_block(self) -> None:
        parsed = extract_code(
            "Here is code:\n```text\nnot python\n```\n```python\ndef add_one(x):\n    return x + 1\n```"
        )

        self.assertTrue(parsed.ok)
        self.assertIn("def add_one", parsed.code)
        self.assertEqual(parsed.parser, "fenced:python:2")

    def test_parser_reports_syntax_error(self) -> None:
        parsed = extract_code("```python\ndef bad(:\n    pass\n```")

        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.status.value, "syntax_error")

    def test_generate_rollouts_with_static_generator_scores_solution(self) -> None:
        records = generate_rollouts(
            [_task()],
            generator=StaticGenerator("```python\ndef add_one(x):\n    return x + 1\n```"),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=1,
            seed=1,
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            include_hidden=True,
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].reward["reward"], 1.0)
        self.assertTrue(records[0].execution["passed"])

    def test_rollout_jsonl_round_trip(self) -> None:
        records = generate_rollouts(
            [_task()],
            generator=MockGenerator(),
            prompt_template=DIRECT_SOLUTION_TEMPLATE,
            samples_per_task=2,
            seed=1,
            max_new_tokens=128,
            temperature=0.8,
            top_p=0.95,
            include_hidden=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rollouts.jsonl"
            write_rollouts_jsonl(records, path)
            restored = read_rollouts_jsonl(path)

        self.assertEqual(len(restored), 2)
        self.assertEqual(restored[0].task_id, "agent/add-one")

    def test_run_rollouts_script_writes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / "tasks.jsonl"
            output_path = Path(tmpdir) / "rollouts.jsonl"
            TaskRegistry([_task()]).to_jsonl(task_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_rollouts.py"),
                    "--tasks",
                    str(task_path),
                    "--output",
                    str(output_path),
                    "--backend",
                    "mock",
                    "--samples-per-task",
                    "2",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output_path.exists())
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(records), 2)
        self.assertIn("mean_reward", completed.stdout)


if __name__ == "__main__":
    unittest.main()
