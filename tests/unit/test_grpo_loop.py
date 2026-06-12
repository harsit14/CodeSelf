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
from codeself.training.grpo_loop import run_grpo_microbatched_training_loop  # noqa: E402


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


class GRPOMicrobatchedLoopTests(unittest.TestCase):
    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_microbatched_loop_matches_full_batch_update(self) -> None:
        import torch

        def make_param() -> "torch.nn.Parameter":
            return torch.nn.Parameter(
                torch.tensor(
                    [
                        [0.0, -1.0, -0.4],
                        [0.0, -0.8, -0.3],
                        [0.0, -0.6, -1.0],
                        [0.0, -0.5, -0.9],
                    ]
                )
            )

        # One 4-sample batch, no KL, no grad clipping: a full-batch update and
        # a microbatch-of-2 accumulated update must reach the same parameters.
        batch = assign_group_relative_advantages(
            TrainingBatch(
                tuple(_sample(sample_index=i, reward=float(i)) for i in range(4))
            )
        ).batch
        config = GRPOTrainingLoopConfig(
            optimizer_step=GRPOOptimizerStepConfig(
                loss=GRPOLossConfig(kl_beta=0.0),
                max_grad_norm=None,
            )
        )

        full_param = make_param()
        full_opt = torch.optim.SGD([full_param], lr=0.1)
        run_grpo_training_loop(
            (batch,),
            tensor_batch_builder=_IndexedParameterBatchBuilder(full_param),
            optimizer=full_opt,
            config=config,
        )

        micro_param = make_param()
        micro_opt = torch.optim.SGD([micro_param], lr=0.1)
        micro_result = run_grpo_microbatched_training_loop(
            (batch,),
            tensor_batch_builder=_IndexedParameterBatchBuilder(micro_param),
            microbatch_size=2,
            optimizer=micro_opt,
            config=config,
        )

        self.assertEqual(micro_result.microbatch_count, 2)
        self.assertEqual(micro_result.optimizer_step_count, 1)
        self.assertEqual([s.accumulation_size for s in micro_result.steps], [2, 2])
        self.assertTrue(
            torch.allclose(full_param.detach(), micro_param.detach(), atol=1e-6),
            msg=f"max diff {(full_param - micro_param).abs().max().item()}",
        )

    def test_microbatched_loop_rejects_nonpositive_size(self) -> None:
        with self.assertRaises(ValueError):
            run_grpo_microbatched_training_loop(
                (),
                tensor_batch_builder=lambda batch: None,
                microbatch_size=0,
                optimizer=object(),
            )


class _IndexedParameterBatchBuilder:
    """Selects each sample's parameter row by its sample_index.

    Unlike `_ParameterBatchBuilder`, this maps samples to rows correctly when a
    batch is split into microbatches, so a microbatched run is comparable to a
    full-batch run.
    """

    def __init__(self, parameter: object) -> None:
        self.parameter = parameter

    def __call__(self, batch: TrainingBatch) -> GRPOTensorBatch:
        import torch

        indices = [sample.sample_index for sample in batch.samples]
        policy_logprobs = self.parameter[indices]
        response_mask = torch.zeros_like(policy_logprobs)
        response_mask[:, 1:] = 1.0
        return GRPOTensorBatch(
            policy_logprobs=policy_logprobs,
            old_policy_logprobs=torch.zeros_like(policy_logprobs),
            response_mask=response_mask,
            advantages=torch.tensor([sample.advantage for sample in batch.samples]),
            reference_logprobs=None,
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
