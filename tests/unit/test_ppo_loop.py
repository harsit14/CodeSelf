from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    PPOLossConfig,
    PPOOptimizerStepConfig,
    PPOTensorBatch,
    PPOTrainingLoopConfig,
    SequenceTrainingSample,
    TrainingBatch,
    build_prompt_response_mask,
    run_ppo_training_loop,
    torch_training_available,
)


class PPOTrainingLoopTests(unittest.TestCase):
    def test_training_loop_config_validates_max_batches(self) -> None:
        self.assertIsNone(PPOTrainingLoopConfig().max_batches)
        with self.assertRaises(ValueError):
            PPOTrainingLoopConfig(max_batches=0)

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_training_loop_accumulates_gradients_and_steps_scheduler(self) -> None:
        import torch

        policy = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            )
        )
        values = torch.nn.Parameter(torch.zeros((3, 3)))
        optimizer = torch.optim.SGD([policy, values], lr=0.05)
        scheduler = _CountingScheduler()
        batches = tuple(_training_batch(index=index, reward=float(index)) for index in range(3))

        before_policy = policy.detach().clone()
        before_values = values.detach().clone()
        result = run_ppo_training_loop(
            batches,
            tensor_batch_builder=_ParameterBatchBuilder(policy, values),
            optimizer=optimizer,
            scheduler=scheduler,
            config=PPOTrainingLoopConfig(
                optimizer_step=PPOOptimizerStepConfig(
                    loss=PPOLossConfig(
                        kl_beta=0.0,
                        normalize_advantages=False,
                        value_clip_epsilon=None,
                    ),
                    gradient_accumulation_steps=2,
                    max_grad_norm=None,
                )
            ),
        )

        self.assertEqual(result.microbatch_count, 3)
        self.assertEqual(result.optimizer_step_count, 2)
        self.assertEqual(scheduler.steps, 2)
        self.assertEqual([step.optimizer_step for step in result.steps], [False, True, True])
        self.assertEqual([step.accumulation_size for step in result.steps], [2, 2, 1])
        self.assertFalse(torch.equal(before_policy, policy.detach()))
        self.assertFalse(torch.equal(before_values, values.detach()))
        self.assertGreater(result.total_response_tokens, 0)
        self.assertGreater(result.mean_value_loss, 0.0)

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_training_loop_respects_max_batches(self) -> None:
        import torch

        policy = torch.nn.Parameter(torch.zeros((2, 3)))
        values = torch.nn.Parameter(torch.zeros((2, 3)))
        optimizer = torch.optim.SGD([policy, values], lr=0.01)
        batches = tuple(_training_batch(index=index, reward=float(index)) for index in range(3))

        result = run_ppo_training_loop(
            batches,
            tensor_batch_builder=_ParameterBatchBuilder(policy, values),
            optimizer=optimizer,
            config=PPOTrainingLoopConfig(
                max_batches=1,
                optimizer_step=PPOOptimizerStepConfig(
                    loss=PPOLossConfig(
                        kl_beta=0.0,
                        normalize_advantages=False,
                        value_clip_epsilon=None,
                    ),
                    max_grad_norm=None,
                ),
            ),
        )

        self.assertEqual(result.microbatch_count, 1)
        self.assertEqual(result.optimizer_step_count, 1)
        self.assertEqual(result.steps[0].step, 1)

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_training_loop_requires_batches(self) -> None:
        import torch

        policy = torch.nn.Parameter(torch.zeros((1, 3)))
        values = torch.nn.Parameter(torch.zeros((1, 3)))
        optimizer = torch.optim.SGD([policy, values], lr=0.01)

        with self.assertRaises(ValueError):
            run_ppo_training_loop(
                (),
                tensor_batch_builder=_ParameterBatchBuilder(policy, values),
                optimizer=optimizer,
            )


class _ParameterBatchBuilder:
    def __init__(self, policy: object, values: object) -> None:
        self.policy = policy
        self.values = values

    def __call__(self, batch: TrainingBatch) -> PPOTensorBatch:
        import torch

        rows = len(batch.samples)
        policy_logprobs = self.policy[:rows]
        value_predictions = self.values[:rows]
        response_mask = torch.zeros_like(policy_logprobs)
        response_mask[:, 1:] = 1.0
        advantages = torch.zeros_like(policy_logprobs)
        returns = torch.zeros_like(policy_logprobs)
        for row, sample in enumerate(batch.samples):
            advantages[row, 1:] = float(sample.reward) + 1.0
            returns[row, 1:] = float(sample.reward) + 1.0
        return PPOTensorBatch(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=torch.zeros_like(policy_logprobs),
            response_mask=response_mask,
            advantages=advantages,
            returns=returns,
            values=value_predictions,
            old_values=torch.zeros_like(value_predictions),
            entropy=torch.zeros_like(policy_logprobs),
            reference_logprobs=None,
        )


class _CountingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _training_batch(*, index: int, reward: float) -> TrainingBatch:
    samples = (
        _sample(sample_index=0, reward=reward),
        _sample(sample_index=1, reward=reward + 1.0),
    )
    return TrainingBatch(samples)


def _sample(*, sample_index: int, reward: float) -> SequenceTrainingSample:
    input_ids = (1, 2, 3)
    return SequenceTrainingSample(
        task_id="task/a",
        sample_index=sample_index,
        input_ids=input_ids,
        masks=build_prompt_response_mask(token_count=3, prompt_token_count=1),
        reward=reward,
    )


if __name__ == "__main__":
    unittest.main()
