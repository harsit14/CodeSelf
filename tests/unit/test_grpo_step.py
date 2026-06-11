from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    GRPOLossConfig,
    GRPOOptimizerStepConfig,
    GRPOTensorBatch,
    SequenceTrainingSample,
    TokenLogprobs,
    TrainingBatch,
    assign_group_relative_advantages,
    build_grpo_tensor_batch,
    build_prompt_response_mask,
    run_grpo_optimizer_step,
    torch_training_available,
)


class GRPOStepTests(unittest.TestCase):
    def test_optimizer_step_config_validates_ranges(self) -> None:
        self.assertEqual(GRPOOptimizerStepConfig().gradient_accumulation_steps, 1)
        with self.assertRaises(ValueError):
            GRPOOptimizerStepConfig(gradient_accumulation_steps=0)
        with self.assertRaises(ValueError):
            GRPOOptimizerStepConfig(max_grad_norm=0.0)

    def test_optimizer_step_reports_missing_torch_cleanly(self) -> None:
        if torch_training_available():
            with self.assertRaises(TypeError):
                run_grpo_optimizer_step(
                    GRPOTensorBatch(
                        policy_logprobs=None,
                        old_policy_logprobs=None,
                        response_mask=None,
                        advantages=None,
                    ),
                    optimizer=_FakeOptimizer(),
                )
        else:
            with self.assertRaisesRegex(RuntimeError, "optional training dependency"):
                run_grpo_optimizer_step(
                    GRPOTensorBatch(
                        policy_logprobs=None,
                        old_policy_logprobs=None,
                        response_mask=None,
                        advantages=None,
                    ),
                    optimizer=_FakeOptimizer(),
                )

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_optimizer_step_backprops_and_updates_parameter(self) -> None:
        import torch

        parameter = torch.nn.Parameter(torch.tensor([[0.0, -1.0, math.log(2.0)]]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        batch = GRPOTensorBatch(
            policy_logprobs=parameter,
            old_policy_logprobs=torch.tensor([[0.0, -1.0, 0.0]]),
            reference_logprobs=torch.tensor([[0.0, -1.5, 0.0]]),
            response_mask=torch.tensor([[0.0, 1.0, 1.0]]),
            advantages=torch.tensor([1.0]),
        )

        before = parameter.detach().clone()
        result = run_grpo_optimizer_step(batch, optimizer=optimizer)

        self.assertTrue(result.optimizer_step)
        self.assertEqual(result.total_response_tokens, 2)
        self.assertIsNotNone(result.grad_norm)
        self.assertLess(result.backward_loss, 0.0)
        self.assertFalse(torch.equal(before, parameter.detach()))

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_tensor_batch_step_can_use_training_batch_bridge(self) -> None:
        import torch

        model_logprobs = torch.nn.Parameter(
            torch.tensor(
                [
                    [0.0, -1.0, -0.3],
                    [0.0, -0.5, -1.0],
                ]
            )
        )
        training_batch = assign_group_relative_advantages(
            TrainingBatch(
                (
                    _sample(sample_index=0, reward=0.0, policy=(0.0, -1.0, -0.3)),
                    _sample(sample_index=1, reward=1.0, policy=(0.0, -0.5, -1.0)),
                )
            )
        ).batch
        tensor_batch = build_grpo_tensor_batch(training_batch)
        tensor_batch = GRPOTensorBatch(
            policy_logprobs=model_logprobs,
            old_policy_logprobs=tensor_batch.old_policy_logprobs,
            response_mask=tensor_batch.response_mask,
            advantages=tensor_batch.advantages,
            reference_logprobs=None,
        )
        optimizer = torch.optim.SGD([model_logprobs], lr=0.05)

        result = run_grpo_optimizer_step(
            tensor_batch,
            optimizer=optimizer,
            config=GRPOOptimizerStepConfig(
                loss=GRPOLossConfig(kl_beta=0.0),
                max_grad_norm=None,
            ),
        )

        self.assertEqual(result.total_response_tokens, 4)
        self.assertTrue(result.optimizer_step)
        self.assertIsNone(result.grad_norm)


class _FakeOptimizer:
    def zero_grad(self, *args: object, **kwargs: object) -> None:
        return None

    def step(self) -> None:
        return None


def _sample(
    *,
    sample_index: int,
    reward: float,
    policy: tuple[float, ...],
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
            old_policy_logprobs=tuple(0.0 for _ in policy),
        ),
    )


if __name__ == "__main__":
    unittest.main()
