from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    PPOLossConfig,
    PPOOptimizerStepConfig,
    PPOValueEstimate,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    build_ppo_tensor_batch_from_model,
    build_prompt_response_mask,
    gather_causal_lm_token_entropy,
    pad_ppo_training_batch_tensors,
    run_ppo_optimizer_step,
    torch_training_available,
)


@unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
class PPOModelForwardTests(unittest.TestCase):
    def test_gather_causal_lm_token_entropy_aligns_next_token_positions(self) -> None:
        import torch

        logits = torch.zeros((1, 3, 4), dtype=torch.float32)
        logits[0, 0, 1] = 3.0
        logits[0, 1, 2] = 1.0
        attention_mask = torch.tensor([[1.0, 1.0, 0.0]])

        entropy = gather_causal_lm_token_entropy(logits, attention_mask)

        expected_first_response = -(
            torch.softmax(logits[0, 0], dim=-1)
            * torch.log_softmax(logits[0, 0], dim=-1)
        ).sum()
        self.assertEqual(tuple(entropy.shape), (1, 3))
        self.assertEqual(float(entropy[0, 0].item()), 0.0)
        self.assertAlmostEqual(
            float(entropy[0, 1].item()),
            expected_first_response.item(),
            places=6,
        )
        self.assertEqual(float(entropy[0, 2].item()), 0.0)

    def test_pad_ppo_training_batch_keeps_recorded_rollout_logprobs(self) -> None:
        batch = TrainingBatch(
            (
                _sample(sample_index=0, reward=0.0, input_ids=(1, 2)),
                _sample(sample_index=1, reward=1.0, input_ids=(1, 3, 4)),
            )
        )

        padded = pad_ppo_training_batch_tensors(batch, pad_token_id=0)

        self.assertEqual(tuple(padded.input_ids.shape), (2, 3))
        self.assertEqual(padded.input_ids.tolist()[0], [1, 2, 0])
        self.assertEqual(padded.response_mask.tolist()[0], [0.0, 1.0, 0.0])
        self.assertIsNotNone(padded.old_policy_logprobs)
        self.assertIsNotNone(padded.reference_logprobs)

    def test_model_tensor_batch_uses_differentiable_policy_and_value_outputs(self) -> None:
        batch = TrainingBatch(
            (
                _sample(sample_index=0, reward=1.0, input_ids=(1, 2, 3)),
                _sample(sample_index=1, reward=0.0, input_ids=(1, 4, 5)),
            )
        )
        policy_model = _ToyPPOCausalLM(vocab_size=8)

        tensor_batch = build_ppo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            config=PPOLossConfig(kl_beta=0.0, normalize_advantages=False),
        )

        self.assertEqual(tuple(tensor_batch.policy_logprobs.shape), (2, 3))
        self.assertEqual(tuple(tensor_batch.values.shape), (2, 3))
        self.assertTrue(tensor_batch.policy_logprobs.requires_grad)
        self.assertTrue(tensor_batch.values.requires_grad)
        self.assertTrue(tensor_batch.entropy.requires_grad)
        self.assertFalse(tensor_batch.old_policy_logprobs.requires_grad)
        self.assertFalse(tensor_batch.old_values.requires_grad)
        self.assertEqual(tensor_batch.response_mask.tolist()[0], [0.0, 1.0, 1.0])

    def test_model_tensor_batch_can_drive_optimizer_updates(self) -> None:
        import torch

        batch = TrainingBatch(
            (
                _sample_without_logprobs(sample_index=0, reward=1.0, input_ids=(1, 2)),
                _sample_without_logprobs(sample_index=1, reward=0.0, input_ids=(1, 3)),
            )
        )
        policy_model = _ToyPPOCausalLM(vocab_size=6)
        optimizer = torch.optim.SGD(policy_model.parameters(), lr=0.1)
        before_logits = policy_model.logit_table.detach().clone()
        before_values = policy_model.value_table.detach().clone()

        tensor_batch = build_ppo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            config=PPOLossConfig(
                kl_beta=0.0,
                normalize_advantages=False,
                value_clip_epsilon=None,
            ),
        )
        result = run_ppo_optimizer_step(
            tensor_batch,
            optimizer=optimizer,
            config=PPOOptimizerStepConfig(
                loss=PPOLossConfig(
                    kl_beta=0.0,
                    normalize_advantages=False,
                    value_clip_epsilon=None,
                ),
                max_grad_norm=None,
            ),
        )

        self.assertEqual(result.total_response_tokens, 2)
        self.assertGreater(result.value_loss, 0.0)
        self.assertFalse(torch.equal(before_logits, policy_model.logit_table.detach()))
        self.assertFalse(torch.equal(before_values, policy_model.value_table.detach()))

    def test_model_tensor_batch_detaches_old_and_reference_models(self) -> None:
        batch = TrainingBatch(
            (_sample(sample_index=0, reward=1.0, input_ids=(1, 2, 3)),)
        )
        policy_model = _ToyPPOCausalLM(vocab_size=8)
        old_policy_model = _ToyPPOCausalLM(vocab_size=8)
        reference_model = _ToyPPOCausalLM(vocab_size=8)
        old_value_model = _ToyValueModel(vocab_size=8)

        tensor_batch = build_ppo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            old_policy_model=old_policy_model,
            old_value_model=old_value_model,
            reference_model=reference_model,
            config=PPOLossConfig(kl_beta=0.1, normalize_advantages=False),
        )

        self.assertFalse(tensor_batch.old_policy_logprobs.requires_grad)
        self.assertFalse(tensor_batch.old_values.requires_grad)
        self.assertIsNotNone(tensor_batch.reference_logprobs)
        self.assertFalse(tensor_batch.reference_logprobs.requires_grad)

    def test_model_tensor_batch_accepts_explicit_value_estimates_for_targets(self) -> None:
        import torch

        batch = TrainingBatch((_sample(sample_index=0, reward=1.0, input_ids=(1, 2)),))
        policy_model = _ToyPPOCausalLM(vocab_size=5)
        estimates = {
            ("task/a", 0): PPOValueEstimate(
                values=(0.0, 0.25),
                old_values=(0.0, 0.25),
                rewards=(0.0, 0.5),
            )
        }

        tensor_batch = build_ppo_tensor_batch_from_model(
            batch,
            policy_model=policy_model,
            value_estimates=estimates,
            config=PPOLossConfig(normalize_advantages=False),
        )

        self.assertTrue(torch.allclose(tensor_batch.returns[0], torch.tensor([0.0, 0.5])))
        self.assertTrue(
            torch.allclose(tensor_batch.old_values[0], torch.tensor([0.0, 0.25]))
        )

    def test_model_tensor_batch_requires_value_outputs(self) -> None:
        batch = TrainingBatch((_sample(sample_index=0, reward=1.0, input_ids=(1, 2)),))

        with self.assertRaises(ValueError):
            build_ppo_tensor_batch_from_model(
                batch,
                policy_model=_ToyLogitsOnlyCausalLM(vocab_size=5),
            )


class _ToyPPOCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))
        self.value_table = torch.nn.Parameter(torch.zeros(vocab_size))

    def parameters(self) -> tuple[object, object]:
        return (self.logit_table, self.value_table)

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return SimpleNamespace(
            logits=self.logit_table[input_ids],
            values=self.value_table[input_ids],
        )


class _ToyValueModel:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.value_table = torch.nn.Parameter(torch.zeros(vocab_size))

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return self.value_table[input_ids]


class _ToyLogitsOnlyCausalLM:
    def __init__(self, *, vocab_size: int) -> None:
        import torch

        self.logit_table = torch.nn.Parameter(torch.zeros((vocab_size, vocab_size)))

    def __call__(self, *, input_ids: object, attention_mask: object) -> object:
        return SimpleNamespace(logits=self.logit_table[input_ids])


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
            reference_logprobs=old_policy_logprobs,
        ),
    )


def _sample_without_logprobs(
    *,
    sample_index: int,
    reward: float,
    input_ids: tuple[int, ...],
) -> SequenceTrainingSample:
    return SequenceTrainingSample(
        task_id="task/a",
        sample_index=sample_index,
        input_ids=input_ids,
        masks=build_prompt_response_mask(
            token_count=len(input_ids),
            prompt_token_count=1,
        ),
        reward=reward,
    )


if __name__ == "__main__":
    unittest.main()
