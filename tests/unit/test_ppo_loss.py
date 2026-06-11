from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    PPOLossConfig,
    PPOValueEstimate,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    build_prompt_response_mask,
    compute_ppo_loss,
    prepare_ppo_value_targets,
)


class PPOLossTests(unittest.TestCase):
    def test_prepare_value_targets_uses_sparse_final_reward_and_gae(self) -> None:
        batch = TrainingBatch((_sample(reward=1.0),))
        estimates = (
            PPOValueEstimate(
                values=(0.0, 0.0, 0.0),
                old_values=(0.0, 0.2, 0.1),
            ),
        )

        result = prepare_ppo_value_targets(
            batch,
            estimates,
            config=PPOLossConfig(normalize_advantages=False),
        )
        target = result.targets[0]

        self.assertEqual(target.response_positions, (1, 2))
        self.assertEqual(target.rewards, (0.0, 1.0))
        self.assertAlmostEqual(target.raw_advantages[0], 0.8)
        self.assertAlmostEqual(target.raw_advantages[1], 0.9)
        self.assertEqual(target.returns, (1.0, 1.0))
        self.assertAlmostEqual(result.mean_return, 1.0)

    def test_compute_ppo_loss_clips_policy_and_value_and_uses_entropy(self) -> None:
        sample = _sample(
            policy=(0.0, math.log(2.0), 0.0),
            old_policy=(0.0, 0.0, 0.0),
            entropy=(0.0, 0.5, 0.5),
        )

        result = compute_ppo_loss(
            TrainingBatch((sample,)),
            (
                PPOValueEstimate(
                    values=(0.0, 1.5, 0.0),
                    old_values=(0.0, 0.0, 0.0),
                    rewards=(0.0, 1.0),
                ),
            ),
            config=PPOLossConfig(
                clip_epsilon=0.2,
                value_clip_epsilon=0.2,
                value_loss_coef=0.5,
                entropy_coef=0.01,
                normalize_advantages=False,
            ),
        )
        sample_loss = result.samples[0]

        self.assertEqual(result.total_response_tokens, 2)
        self.assertAlmostEqual(sample_loss.mean_ratio, 1.5)
        self.assertAlmostEqual(sample_loss.mean_clipped_ratio, 1.1)
        self.assertAlmostEqual(sample_loss.policy_loss, -1.1)
        self.assertAlmostEqual(sample_loss.value_loss, 0.41)
        self.assertAlmostEqual(sample_loss.entropy_loss, -0.005)
        self.assertAlmostEqual(sample_loss.total_loss, -0.695)
        self.assertAlmostEqual(result.clipped_policy_fraction, 0.5)
        self.assertAlmostEqual(result.clipped_value_fraction, 0.5)

    def test_compute_ppo_loss_adds_reference_kl_penalty(self) -> None:
        sample = _sample(
            policy=(0.0, -1.0),
            old_policy=(0.0, -1.0),
            reference=(0.0, -2.0),
            reward=0.0,
            prompt_tokens=1,
        )
        expected_kl = math.exp(-1.0) - (-1.0) - 1.0

        result = compute_ppo_loss(
            TrainingBatch((sample,)),
            {("task/a", 0): PPOValueEstimate(values=(0.0, 0.0), old_values=(0.0, 0.0))},
            config=PPOLossConfig(kl_beta=0.5, normalize_advantages=False),
        )

        self.assertAlmostEqual(result.samples[0].mean_approx_kl, expected_kl)
        self.assertAlmostEqual(result.samples[0].kl_loss, expected_kl * 0.5)
        self.assertAlmostEqual(result.mean_kl_loss, expected_kl * 0.5)

    def test_prepare_value_targets_normalizes_advantages_across_batch(self) -> None:
        batch = TrainingBatch(
            (
                _sample(task_id="task/a", sample_index=0, reward=0.0),
                _sample(task_id="task/a", sample_index=1, reward=2.0),
            )
        )
        estimates = (
            PPOValueEstimate(values=(0.0, 0.0, 0.0), old_values=(0.0, 0.0, 0.0)),
            PPOValueEstimate(values=(0.0, 0.0, 0.0), old_values=(0.0, 0.0, 0.0)),
        )

        result = prepare_ppo_value_targets(batch, estimates)
        advantages = tuple(value for target in result.targets for value in target.advantages)

        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertAlmostEqual(max(advantages), 1.0)
        self.assertAlmostEqual(min(advantages), -1.0)

    def test_compute_ppo_loss_validates_required_fields(self) -> None:
        base = _sample()
        without_logprobs = replace(base, logprobs=None)
        without_old = replace(
            base,
            logprobs=TokenLogprobs(
                token_ids=base.input_ids,
                policy_logprobs=base.logprobs.policy_logprobs,
            ),
        )
        estimates = (PPOValueEstimate(values=(0.0, 0.0, 0.0)),)

        with self.assertRaises(ValueError):
            compute_ppo_loss(TrainingBatch((without_logprobs,)), estimates)
        with self.assertRaises(ValueError):
            compute_ppo_loss(TrainingBatch((without_old,)), estimates)
        with self.assertRaises(ValueError):
            prepare_ppo_value_targets(
                TrainingBatch((base,)),
                (PPOValueEstimate(values=(0.0, 0.0)),),
            )
        with self.assertRaises(ValueError):
            compute_ppo_loss(
                TrainingBatch((base,)),
                estimates,
                config=PPOLossConfig(kl_beta=0.1),
            )
        with self.assertRaises(ValueError):
            PPOLossConfig(gamma=0.0)


def _sample(
    *,
    task_id: str = "task/a",
    sample_index: int = 0,
    reward: float = 1.0,
    policy: tuple[float, ...] = (0.0, -1.0, -1.0),
    old_policy: tuple[float, ...] | None = (0.0, -1.0, -1.0),
    reference: tuple[float, ...] | None = None,
    entropy: tuple[float, ...] | None = None,
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
        entropy=entropy,
    )
    return SequenceTrainingSample(
        task_id=task_id,
        sample_index=sample_index,
        input_ids=input_ids,
        masks=masks,
        reward=reward,
        logprobs=logprobs,
    )


if __name__ == "__main__":
    unittest.main()
