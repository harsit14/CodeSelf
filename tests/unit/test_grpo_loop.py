from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    GRPOLossConfig,
    GRPOOptimizerStepConfig,
    GRPOTensorBatch,
    GRPOTrainingLoopConfig,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    assign_group_relative_advantages,
    build_prompt_response_mask,
    run_grpo_training_loop,
    torch_training_available,
)


class GRPOTrainingLoopTests(unittest.TestCase):
    def test_training_loop_config_validates_max_batches(self) -> None:
        self.assertIsNone(GRPOTrainingLoopConfig().max_batches)
        with self.assertRaises(ValueError):
            GRPOTrainingLoopConfig(max_batches=0)

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_training_loop_accumulates_gradients_and_steps_scheduler(self) -> None:
        import torch

        parameter = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, -1.0, -0.4],
                    [0.0, -0.8, -0.3],
                    [0.0, -0.6, -1.0],
                ]
            )
        )
        optimizer = torch.optim.SGD([parameter], lr=0.05)
        scheduler = _CountingScheduler()
        batches = tuple(_training_batch(index=index, reward=float(index)) for index in range(3))

        before = parameter.detach().clone()
        result = run_grpo_training_loop(
            batches,
            tensor_batch_builder=_ParameterBatchBuilder(parameter),
            optimizer=optimizer,
            scheduler=scheduler,
            config=GRPOTrainingLoopConfig(
                optimizer_step=GRPOOptimizerStepConfig(
                    loss=GRPOLossConfig(kl_beta=0.0),
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
        self.assertFalse(torch.equal(before, parameter.detach()))
        self.assertGreater(result.total_response_tokens, 0)

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_training_loop_respects_max_batches(self) -> None:
        import torch

        parameter = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, -1.0, -0.4],
                    [0.0, -0.8, -0.3],
                ]
            )
        )
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        batches = tuple(_training_batch(index=index, reward=float(index)) for index in range(3))

        result = run_grpo_training_loop(
            batches,
            tensor_batch_builder=_ParameterBatchBuilder(parameter),
            optimizer=optimizer,
            config=GRPOTrainingLoopConfig(
                max_batches=1,
                optimizer_step=GRPOOptimizerStepConfig(
                    loss=GRPOLossConfig(kl_beta=0.0),
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

        parameter = torch.nn.Parameter(torch.tensor([[0.0, -1.0, -0.4]]))
        optimizer = torch.optim.SGD([parameter], lr=0.01)

        with self.assertRaises(ValueError):
            run_grpo_training_loop(
                (),
                tensor_batch_builder=_ParameterBatchBuilder(parameter),
                optimizer=optimizer,
            )


class _ParameterBatchBuilder:
    def __init__(self, parameter: object) -> None:
        self.parameter = parameter

    def __call__(self, batch: TrainingBatch) -> GRPOTensorBatch:
        import torch

        rows = len(batch.samples)
        policy_logprobs = self.parameter[:rows]
        response_mask = torch.zeros_like(policy_logprobs)
        response_mask[:, 1:] = 1.0
        return GRPOTensorBatch(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=torch.zeros_like(policy_logprobs),
            response_mask=response_mask,
            advantages=torch.tensor([sample.advantage for sample in batch.samples]),
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
    return assign_group_relative_advantages(TrainingBatch(samples)).batch


def _sample(*, sample_index: int, reward: float) -> SequenceTrainingSample:
    input_ids = (1, 2, 3)
    masks = build_prompt_response_mask(token_count=3, prompt_token_count=1)
    return SequenceTrainingSample(
        task_id="task/a",
        sample_index=sample_index,
        input_ids=input_ids,
        masks=masks,
        reward=reward,
        logprobs=TokenLogprobs(
            token_ids=input_ids,
            policy_logprobs=(0.0, -1.0, -0.5),
            old_policy_logprobs=(0.0, 0.0, 0.0),
        ),
    )


if __name__ == "__main__":
    unittest.main()
