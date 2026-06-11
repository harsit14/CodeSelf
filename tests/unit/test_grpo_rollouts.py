from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import ParsedCompletion, ParseStatus, RolloutRecord, extract_code  # noqa: E402
from codeself.training import (  # noqa: E402
    GRPORolloutBatchConfig,
    WhitespaceTokenizerEngine,
    build_grpo_training_batch_from_rollouts,
)


class GRPORolloutBatchTests(unittest.TestCase):
    def test_builds_batch_from_raw_completion_and_assigns_advantages(self) -> None:
        tokenizer = WhitespaceTokenizerEngine()
        records = (
            _record(
                task_id="task/add-one",
                sample_index=0,
                reward=0.0,
                raw_completion="```python\ndef add_one(x):\n    return x\n```",
            ),
            _record(
                task_id="task/add-one",
                sample_index=1,
                reward=1.0,
                raw_completion="```python\ndef add_one(x):\n    return x + 1\n```",
            ),
        )

        result = build_grpo_training_batch_from_rollouts(
            records,
            tokenizer,
            config=GRPORolloutBatchConfig(max_prompt_tokens=3, max_response_tokens=8),
        )

        self.assertEqual(result.records_seen, 2)
        self.assertEqual(result.records_used, 2)
        self.assertEqual(result.skipped, ())
        self.assertEqual(result.batch.size, 2)
        self.assertEqual(result.batch.total_response_tokens, 14)
        self.assertGreater(result.batch.samples[0].response_tokens, 0)
        self.assertLessEqual(result.batch.samples[1].response_tokens, 8)
        self.assertAlmostEqual(result.batch.samples[0].advantage or 0.0, -1.0)
        self.assertAlmostEqual(result.batch.samples[1].advantage or 0.0, 1.0)
        self.assertEqual(result.groups[0].sample_indices, (0, 1))
        self.assertEqual(result.batch.samples[0].metadata["response_source"], "raw_completion")
        self.assertTrue(result.batch.samples[0].metadata["prompt_truncated"])
        self.assertFalse(result.batch.samples[0].metadata["execution_passed"])
        self.assertEqual(result.batch.samples[1].metadata["rollout_reward_mode"], "binary")

    def test_can_train_on_parsed_code_and_skip_failed_parses(self) -> None:
        tokenizer = WhitespaceTokenizerEngine()
        records = (
            _record(
                task_id="task/parsed",
                sample_index=0,
                reward=1.0,
                raw_completion="Here:\n```python\ndef solve():\n    return 1\n```",
            ),
            _record(
                task_id="task/parsed",
                sample_index=1,
                reward=0.0,
                raw_completion="```python\ndef broken(:\n    pass\n```",
            ),
        )

        result = build_grpo_training_batch_from_rollouts(
            records,
            tokenizer,
            config=GRPORolloutBatchConfig(
                response_source="parsed_code",
                include_failed_parses=False,
            ),
        )

        sample = result.batch.samples[0]
        response_text = tokenizer.decode(sample.input_ids[-sample.response_tokens :])
        self.assertEqual(result.records_seen, 2)
        self.assertEqual(result.records_used, 1)
        self.assertEqual(result.skipped[0].reason, "parse_status_syntax_error")
        self.assertEqual(sample.advantage, 0.0)
        self.assertEqual(sample.metadata["response_source"], "parsed_code")
        self.assertEqual(sample.metadata["parse_status"], "ok")
        self.assertIn("def", response_text)
        self.assertNotIn("Here:", response_text)

    def test_empty_response_rollouts_are_reported_and_rejected(self) -> None:
        tokenizer = WhitespaceTokenizerEngine()

        with self.assertRaisesRegex(ValueError, "no rollout records"):
            build_grpo_training_batch_from_rollouts(
                (
                    _record(
                        task_id="task/empty",
                        sample_index=0,
                        reward=0.0,
                        raw_completion="   ",
                        parsed=ParsedCompletion(
                            raw_text="   ",
                            code="",
                            status=ParseStatus.EMPTY,
                            parser="empty",
                        ),
                    ),
                ),
                tokenizer,
            )

    def test_config_validates_ranges_and_response_source(self) -> None:
        with self.assertRaises(ValueError):
            GRPORolloutBatchConfig(max_prompt_tokens=0)
        with self.assertRaises(ValueError):
            GRPORolloutBatchConfig(max_response_tokens=-1)
        with self.assertRaises(ValueError):
            GRPORolloutBatchConfig(response_source="completion")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GRPORolloutBatchConfig(min_reward_std=-0.1)


def _record(
    *,
    task_id: str,
    sample_index: int,
    reward: float,
    raw_completion: str,
    parsed: ParsedCompletion | None = None,
) -> RolloutRecord:
    parsed_completion = parsed or extract_code(raw_completion)
    return RolloutRecord(
        task_id=task_id,
        sample_index=sample_index,
        prompt_template="direct_solution_v1",
        prompt="Write a Python solution for this task with a concise function.",
        raw_completion=raw_completion,
        parsed=parsed_completion,
        backend="static",
        model_name="unit-test-model",
        generation_metadata={"temperature": 0.2},
        execution={"passed": reward >= 1.0},
        reward={
            "reward": reward,
            "reward_name": "reward_binary_all_tests_pass",
        },
        metadata={"reward_mode": "binary"},
    )


if __name__ == "__main__":
    unittest.main()
