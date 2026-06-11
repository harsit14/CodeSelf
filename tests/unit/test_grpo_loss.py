from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    GRPOLossConfig,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    assign_group_relative_advantages,
    build_prompt_response_mask,
    compute_grpo_loss,
)


class GRPOLossTests(unittest.TestCase):
    def test_assign_group_relative_advantages_normalizes_by_task(self) -> None:
        batch = TrainingBatch(
            (
                _sample(task_id="task/a", sample_index=1, reward=3.0),
                _sample(task_id="task/b", sample_index=0, reward=5.0),
                _sample(task_id="task/a", sample_index=0, reward=1.0),
            )
        )

        result = assign_group_relative_advantages(batch)

        self.assertEqual([group.task_id for group in result.groups], ["task/a", "task/b"])
        self.assertEqual(result.groups[0].sample_indices, (0, 1))
        self.assertEqual(result.groups[0].advantages, (-1.0, 1.0))
        self.assertEqual(result.groups[1].advantages, (0.0,))
        self.assertEqual(result.batch.samples[0].advantage, 1.0)
        self.assertEqual(result.batch.samples[1].advantage, 0.0)
        self.assertEqual(result.batch.samples[2].advantage, -1.0)

    def test_compute_grpo_loss_uses_response_tokens_only(self) -> None:
        sample = _sample(
            advantage=2.0,
            policy=(10.0, -1.0, -2.0),
            old_policy=(0.0, -1.0, -2.0),
        )
        result = compute_grpo_loss(
            TrainingBatch((sample,)),
            config=GRPOLossConfig(kl_beta=0.0),
        )

        self.assertEqual(result.total_response_tokens, 2)
        self.assertAlmostEqual(result.mean_ratio, 1.0)
        self.assertAlmostEqual(result.mean_policy_loss, -2.0)
        self.assertAlmostEqual(result.mean_loss, -2.0)

    def test_compute_grpo_loss_clips_positive_advantage(self) -> None:
        sample = _sample(
            advantage=1.0,
            policy=(0.0, math.log(2.0)),
            old_policy=(0.0, 0.0),
            prompt_tokens=1,
        )

        result = compute_grpo_loss(
            TrainingBatch((sample,)),
            config=GRPOLossConfig(clip_epsilon=0.2, kl_beta=0.0),
        )

        self.assertAlmostEqual(result.samples[0].mean_ratio, 2.0)
        self.assertAlmostEqual(result.samples[0].mean_clipped_ratio, 1.2)
        self.assertAlmostEqual(result.samples[0].policy_loss, -1.2)
        self.assertAlmostEqual(result.clipped_token_fraction, 1.0)

    def test_compute_grpo_loss_clips_negative_advantage_conservatively(self) -> None:
        sample = _sample(
            advantage=-1.0,
            policy=(0.0, math.log(0.5)),
            old_policy=(0.0, 0.0),
            prompt_tokens=1,
        )

        result = compute_grpo_loss(
            TrainingBatch((sample,)),
            config=GRPOLossConfig(clip_epsilon=0.2, kl_beta=0.0),
        )

        self.assertAlmostEqual(result.samples[0].mean_ratio, 0.5)
        self.assertAlmostEqual(result.samples[0].mean_clipped_ratio, 0.8)
        self.assertAlmostEqual(result.samples[0].policy_loss, 0.8)
        self.assertAlmostEqual(result.clipped_token_fraction, 1.0)

    def test_compute_grpo_loss_adds_reference_kl_penalty(self) -> None:
        sample = _sample(
            advantage=0.0,
            policy=(0.0, -1.0),
            old_policy=(0.0, -1.0),
            reference=(0.0, -2.0),
            prompt_tokens=1,
        )
        expected_kl = math.exp(-1.0) - (-1.0) - 1.0

        result = compute_grpo_loss(
            TrainingBatch((sample,)),
            config=GRPOLossConfig(kl_beta=0.5),
        )

        self.assertAlmostEqual(result.samples[0].mean_approx_kl, expected_kl)
        self.assertAlmostEqual(result.samples[0].kl_loss, expected_kl * 0.5)
        self.assertAlmostEqual(result.mean_loss, expected_kl * 0.5)

    def test_compute_grpo_loss_validates_required_logprob_fields(self) -> None:
        base = _sample(advantage=1.0)
        without_logprobs = replace(base, logprobs=None)
        without_advantage = replace(base, advantage=None)
        without_old = replace(
            base,
            logprobs=TokenLogprobs(
                token_ids=base.input_ids,
                policy_logprobs=base.logprobs.policy_logprobs,
            ),
        )

        with self.assertRaises(ValueError):
            compute_grpo_loss(TrainingBatch((without_logprobs,)))
        with self.assertRaises(ValueError):
            compute_grpo_loss(TrainingBatch((without_advantage,)))
        with self.assertRaises(ValueError):
            compute_grpo_loss(TrainingBatch((without_old,)), config=GRPOLossConfig(kl_beta=0.0))
        with self.assertRaises(ValueError):
            compute_grpo_loss(TrainingBatch((base,)), config=GRPOLossConfig(kl_beta=0.1))


def _sample(
    *,
    task_id: str = "task/a",
    sample_index: int = 0,
    reward: float = 1.0,
    advantage: float | None = None,
    policy: tuple[float, ...] = (0.0, -1.0, -1.0),
    old_policy: tuple[float, ...] | None = (0.0, -1.0, -1.0),
    reference: tuple[float, ...] | None = None,
    prompt_tokens: int = 1,
) -> SequenceTrainingSample:
    input_ids = tuple(range(1, len(policy) + 1))
    masks = build_prompt_response_mask(
        token_count=len(input_ids),
        prompt_token_count=prompt_tokens,
    )
    logprobs = TokenLogprobs(
        token_ids=input_ids,
        policy_logprobs=policy,
        old_policy_logprobs=old_policy,
        reference_logprobs=reference,
    )
    return SequenceTrainingSample(
        task_id=task_id,
        sample_index=sample_index,
        input_ids=input_ids,
        masks=masks,
        reward=reward,
        logprobs=logprobs,
        advantage=advantage,
    )


if __name__ == "__main__":
    unittest.main()
