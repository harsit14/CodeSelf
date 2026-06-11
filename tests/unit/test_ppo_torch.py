from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    PPOLossConfig,
    PPOTensorBatch,
    PPOValueEstimate,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    build_ppo_tensor_batch,
    build_prompt_response_mask,
    compute_ppo_loss,
    compute_ppo_tensor_loss,
    torch_training_available,
)


class PPOTorchTests(unittest.TestCase):
    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_tensor_loss_matches_reference_loss(self) -> None:
        config = PPOLossConfig(
            clip_epsilon=0.2,
            value_clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            kl_beta=0.3,
            normalize_advantages=False,
        )
        batch = TrainingBatch(
            (
                _sample(
                    sample_index=0,
                    reward=1.0,
                    policy=(0.0, math.log(2.0), 0.0),
                    old_policy=(0.0, 0.0, 0.0),
                    reference=(0.0, 0.0, -0.5),
                    entropy=(0.0, 0.5, 0.3),
                ),
                _sample(
                    sample_index=1,
                    reward=0.5,
                    policy=(0.0, -0.5, math.log(0.5), -1.0),
                    old_policy=(0.0, -0.5, 0.0, -1.0),
                    reference=(0.0, -0.25, -0.5, -1.2),
                    entropy=(0.0, 0.2, 0.4, 0.6),
                ),
            )
        )
        estimates = (
            PPOValueEstimate(
                values=(0.0, 1.5, 0.0),
                old_values=(0.0, 0.0, 0.0),
                rewards=(0.0, 1.0),
            ),
            PPOValueEstimate(
                values=(0.0, 0.2, 0.7, 0.0),
                old_values=(0.0, 0.1, 0.1, 0.0),
                rewards=(0.0, 0.0, 0.5),
            ),
        )
        reference = compute_ppo_loss(batch, estimates, config=config)

        tensor_batch = build_ppo_tensor_batch(batch, estimates, config=config)
        tensor_result = compute_ppo_tensor_loss(tensor_batch, config=config)
        payload = tensor_result.to_dict()

        self.assertEqual(tensor_result.total_response_tokens, reference.total_response_tokens)
        self.assertAlmostEqual(payload["loss"], reference.mean_loss, places=6)
        self.assertAlmostEqual(payload["policy_loss"], reference.mean_policy_loss, places=6)
        self.assertAlmostEqual(payload["value_loss"], reference.mean_value_loss, places=6)
        self.assertAlmostEqual(payload["entropy_loss"], reference.mean_entropy_loss, places=6)
        self.assertAlmostEqual(payload["kl_loss"], reference.mean_kl_loss, places=6)
        self.assertAlmostEqual(
            tensor_result.clipped_policy_fraction,
            reference.clipped_policy_fraction,
            places=6,
        )
        self.assertAlmostEqual(
            tensor_result.clipped_value_fraction,
            reference.clipped_value_fraction,
            places=6,
        )

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_tensor_batch_builder_pads_variable_length_samples(self) -> None:
        batch = TrainingBatch(
            (
                _sample(sample_index=0, reward=0.0, policy=(0.0, -1.0)),
                _sample(sample_index=1, reward=1.0, policy=(0.0, -1.0, -2.0)),
            )
        )
        estimates = (
            PPOValueEstimate(values=(0.0, 0.0), old_values=(0.0, 0.0)),
            PPOValueEstimate(values=(0.0, 0.0, 0.0), old_values=(0.0, 0.0, 0.0)),
        )

        tensor_batch = build_ppo_tensor_batch(batch, estimates)

        self.assertEqual(tuple(tensor_batch.policy_logprobs.shape), (2, 3))
        self.assertEqual(tuple(tensor_batch.response_mask.shape), (2, 3))
        self.assertEqual(float(tensor_batch.response_mask[0, -1].item()), 0.0)
        self.assertEqual(float(tensor_batch.response_mask[1, -1].item()), 1.0)
        self.assertEqual(float(tensor_batch.returns[0, -1].item()), 0.0)

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_tensor_loss_is_differentiable_for_policy_and_value_terms(self) -> None:
        import torch

        policy = torch.tensor([[0.0, math.log(1.1)]], requires_grad=True)
        values = torch.tensor([[0.0, 0.5]], requires_grad=True)
        tensor_batch = PPOTensorBatch(
            policy_logprobs=policy,
            old_policy_logprobs=torch.zeros((1, 2)),
            response_mask=torch.tensor([[0.0, 1.0]]),
            advantages=torch.tensor([[0.0, 1.0]]),
            returns=torch.tensor([[0.0, 1.0]]),
            values=values,
            old_values=torch.zeros((1, 2)),
            entropy=torch.zeros((1, 2)),
        )

        result = compute_ppo_tensor_loss(
            tensor_batch,
            config=PPOLossConfig(
                kl_beta=0.0,
                normalize_advantages=False,
                value_clip_epsilon=None,
            ),
        )
        result.loss.backward()

        self.assertIsNotNone(policy.grad)
        self.assertIsNotNone(values.grad)
        self.assertGreater(abs(float(policy.grad[0, 1].item())), 0.0)
        self.assertGreater(abs(float(values.grad[0, 1].item())), 0.0)


def _sample(
    *,
    sample_index: int,
    reward: float,
    policy: tuple[float, ...],
    old_policy: tuple[float, ...] | None = None,
    reference: tuple[float, ...] | None = None,
    entropy: tuple[float, ...] | None = None,
) -> SequenceTrainingSample:
    input_ids = tuple(range(1, len(policy) + 1))
    masks = build_prompt_response_mask(
        token_count=len(input_ids),
        prompt_token_count=1,
    )
    return SequenceTrainingSample(
        task_id="task/a",
        sample_index=sample_index,
        input_ids=input_ids,
        masks=masks,
        reward=reward,
        logprobs=TokenLogprobs(
            token_ids=input_ids,
            policy_logprobs=policy,
            old_policy_logprobs=old_policy or tuple(0.0 for _ in policy),
            reference_logprobs=reference,
            entropy=entropy,
        ),
    )


if __name__ == "__main__":
    unittest.main()
