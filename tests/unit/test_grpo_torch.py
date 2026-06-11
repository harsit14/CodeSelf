from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    GRPOLossConfig,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    assign_group_relative_advantages,
    build_grpo_tensor_batch,
    build_prompt_response_mask,
    compute_grpo_loss,
    compute_grpo_tensor_loss,
    require_torch,
    torch_training_available,
)


class GRPOTorchTests(unittest.TestCase):
    def test_torch_training_availability_is_lazy_boolean(self) -> None:
        self.assertIsInstance(torch_training_available(), bool)

    def test_require_torch_reports_missing_optional_dependency(self) -> None:
        if torch_training_available():
            self.assertEqual(require_torch().__name__, "torch")
        else:
            with self.assertRaisesRegex(RuntimeError, "optional training dependency"):
                require_torch()

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_tensor_loss_matches_reference_loss(self) -> None:
        config = GRPOLossConfig(clip_epsilon=0.2, kl_beta=0.5)
        batch = assign_group_relative_advantages(
            TrainingBatch(
                (
                    _sample(
                        sample_index=0,
                        reward=1.0,
                        policy=(0.0, -1.0, math.log(2.0)),
                        old_policy=(0.0, -1.0, 0.0),
                        reference=(0.0, -1.5, 0.0),
                    ),
                    _sample(
                        sample_index=1,
                        reward=3.0,
                        policy=(0.0, -0.5, math.log(0.5)),
                        old_policy=(0.0, -1.0, 0.0),
                        reference=(0.0, -1.0, -1.0),
                    ),
                )
            )
        ).batch
        reference = compute_grpo_loss(batch, config=config)

        tensor_batch = build_grpo_tensor_batch(batch)
        tensor_result = compute_grpo_tensor_loss(tensor_batch, config=config)
        payload = tensor_result.to_dict()

        self.assertEqual(tensor_result.total_response_tokens, reference.total_response_tokens)
        self.assertAlmostEqual(payload["loss"], reference.mean_loss, places=6)
        self.assertAlmostEqual(payload["policy_loss"], reference.mean_policy_loss, places=6)
        self.assertAlmostEqual(payload["kl_loss"], reference.mean_kl_loss, places=6)
        self.assertAlmostEqual(tensor_result.mean_ratio, reference.mean_ratio, places=6)
        self.assertAlmostEqual(
            tensor_result.clipped_token_fraction,
            reference.clipped_token_fraction,
            places=6,
        )

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_tensor_batch_builder_pads_variable_length_samples(self) -> None:
        batch = assign_group_relative_advantages(
            TrainingBatch(
                (
                    _sample(sample_index=0, reward=0.0, policy=(0.0, -1.0)),
                    _sample(sample_index=1, reward=1.0, policy=(0.0, -1.0, -2.0)),
                )
            )
        ).batch

        tensor_batch = build_grpo_tensor_batch(batch)

        self.assertEqual(tuple(tensor_batch.policy_logprobs.shape), (2, 3))
        self.assertEqual(tuple(tensor_batch.response_mask.shape), (2, 3))
        self.assertEqual(float(tensor_batch.response_mask[0, -1].item()), 0.0)
        self.assertEqual(float(tensor_batch.response_mask[1, -1].item()), 1.0)


def _sample(
    *,
    sample_index: int,
    reward: float,
    policy: tuple[float, ...],
    old_policy: tuple[float, ...] | None = None,
    reference: tuple[float, ...] | None = None,
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
        ),
    )


if __name__ == "__main__":
    unittest.main()
