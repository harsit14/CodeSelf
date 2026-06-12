from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    GRPOLossConfig,
    GRPOOptimizerStepConfig,
    GRPOTrainingLoopConfig,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    assign_group_relative_advantages,
    build_grpo_tensor_batch_from_model,
    build_prompt_response_mask,
    gather_causal_lm_token_logprobs,
    pad_training_batch_tensors,
    run_grpo_training_loop,
    torch_training_available,
)


@unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
class GRPOModelForwardTests(unittest.TestCase):
    def test_gather_causal_lm_token_logprobs_aligns_next_token_predictions(self) -> None:
        import torch

        logits = torch.zeros((1, 3, 5), dtype=torch.float32)
        logits[0, 0, 2] = 2.0
        logits[0, 1, 3] = 4.0
        input_ids = torch.tensor([[1, 2, 3]])

        logprobs = gather_causal_lm_token_logprobs(logits, input_ids)

        self.assertEqual(tuple(logprobs.shape), (1, 3))
        self.assertEqual(float(logprobs[0, 0].item()), 0.0)
        # The memory-efficient gather (selected - logsumexp) is mathematically
        # equal to log_softmax().gather() but differs in float32 rounding at
        # ~1e-7, so compare at float-precision tolerance.
        self.assertAlmostEqual(
            float(logprobs[0, 1].item()),
            torch.log_softmax(logits[0, 0], dim=-1)[2].item(),
            places=5,
        )
        self.assertAlmostEqual(
            float(logprobs[0, 2].item()),
            torch.log_softmax(logits[0, 1], dim=-1)[3].item(),
            places=5,
        )

    def test_pad_training_batch_tensors_pads_inputs_and_recorded_logprobs(self) -> None:
        batch = assign_group_relative_advantages(
            TrainingBatch(
                (
                    _sample(sample_index=0, reward=0.0, input_ids=(1, 2, 3)),
                    _sample(sample_index=1, reward=1.0, input_ids=(1, 4, 5, 6)),
                )
            )
        ).batch

        padded = pad_training_batch_tensors(batch, pad_token_id=0)

        self.assertEqual(tuple(padded.input_ids.shape), (2, 4))
        self.assertEqual(padded.input_ids.tolist()[0], [1, 2, 3, 0])
        self.assertEqual(padded.attention_mask.tolist()[0], [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(padded.response_mask.tolist()[0], [0.0, 1.0, 1.0, 0.0])
        self.assertIsNotNone(padded.old_policy_logprobs)
        self.assertEqual(tuple(padded.advantages.shape), (2,))

    def test_model_tensor_batch_uses_differentiable_policy_and_detached_old_reference(self) -> None:
        import torch

        batch = assign_group_relative_advantages(
            TrainingBatch(
                (
                    _sample(sample_index=0, reward=0.0, input_ids=(1, 2, 3)),
                    _sample(sample_index=1, reward=1.0, input_ids=(1, 4, 5)),
                )
            )
        ).batch
        policy_model = _ToyCausalLM(vocab_size=8)
        reference_model = _ToyCausalLM(vocab_size=8)

        tensor_batch = build_grpo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            reference_model=reference_model,
        )

        self.assertEqual(tuple(tensor_batch.policy_logprobs.shape), (2, 3))
        self.assertTrue(tensor_batch.policy_logprobs.requires_grad)
        self.assertFalse(tensor_batch.old_policy_logprobs.requires_grad)
        self.assertIsNotNone(tensor_batch.reference_logprobs)
        self.assertFalse(tensor_batch.reference_logprobs.requires_grad)
        self.assertEqual(tensor_batch.response_mask.tolist()[0], [0.0, 1.0, 1.0])

    def test_model_tensor_batch_can_drive_training_loop(self) -> None:
        import torch

        policy_model = _ToyCausalLM(vocab_size=8)
        optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.05)
        batches = (
            _training_batch(input_ids_a=(1, 2, 3), input_ids_b=(1, 4, 5)),
            _training_batch(input_ids_a=(1, 2, 6), input_ids_b=(1, 4, 7)),
        )

        before = policy_model.logit_table.detach().clone()
        result = run_grpo_training_loop(
            batches,
            tensor_batch_builder=lambda batch: build_grpo_tensor_batch_from_model(
                batch,
                policy_model=policy_model,
            ),
            optimizer=optimizer,
            config=GRPOTrainingLoopConfig(
                optimizer_step=GRPOOptimizerStepConfig(
                    loss=GRPOLossConfig(kl_beta=0.0),
                    max_grad_norm=None,
                )
            ),
        )

        self.assertEqual(result.optimizer_step_count, 2)
        self.assertFalse(torch.equal(before, policy_model.logit_table.detach()))

    def test_pad_training_batch_rejects_mixed_old_logprobs(self) -> None:
        first = _sample(sample_index=0, reward=0.0, input_ids=(1, 2, 3))
        second = SequenceTrainingSample(
            task_id="task/a",
            sample_index=1,
            input_ids=(1, 4, 5),
            masks=build_prompt_response_mask(token_count=3, prompt_token_count=1),
            reward=1.0,
            logprobs=TokenLogprobs(
                token_ids=(1, 4, 5),
                policy_logprobs=(0.0, -1.0, -1.0),
            ),
            advantage=1.0,
        )

        with self.assertRaises(ValueError):
            pad_training_batch_tensors(TrainingBatch((first, second)))


class _ToyCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))

    def parameters(self) -> tuple[object]:
        return (self.logit_table,)

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return SimpleNamespace(logits=self.logit_table[input_ids])


def _training_batch(
    *,
    input_ids_a: tuple[int, ...],
    input_ids_b: tuple[int, ...],
) -> TrainingBatch:
    return assign_group_relative_advantages(
        TrainingBatch(
            (
                _sample(sample_index=0, reward=0.0, input_ids=input_ids_a),
                _sample(sample_index=1, reward=1.0, input_ids=input_ids_b),
            )
        )
    ).batch


def _sample(
    *,
    sample_index: int,
    reward: float,
    input_ids: tuple[int, ...],
) -> SequenceTrainingSample:
    masks = build_prompt_response_mask(
        token_count=len(input_ids),
        prompt_token_count=1,
    )
    old_policy_logprobs = (0.0,) + tuple(-math.log(8.0) for _ in input_ids[1:])
    return SequenceTrainingSample(
        task_id="task/a",
        sample_index=sample_index,
        input_ids=input_ids,
        masks=masks,
        reward=reward,
        logprobs=TokenLogprobs(
            token_ids=input_ids,
            policy_logprobs=old_policy_logprobs,
            old_policy_logprobs=old_policy_logprobs,
        ),
    )


if __name__ == "__main__":
    unittest.main()
