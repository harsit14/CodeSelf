from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from codeself.training import (  # noqa: E402
    PPOLossConfig,
    PPOOptimizerStepConfig,
    PPOTensorBatch,
    run_ppo_optimizer_step,
    torch_training_available,
)


class PPOStepTests(unittest.TestCase):
    def test_optimizer_step_config_validates_ranges(self) -> None:
        self.assertEqual(PPOOptimizerStepConfig().gradient_accumulation_steps, 1)
        with self.assertRaises(ValueError):
            PPOOptimizerStepConfig(gradient_accumulation_steps=0)
        with self.assertRaises(ValueError):
            PPOOptimizerStepConfig(max_grad_norm=0.0)

    def test_optimizer_step_reports_missing_torch_cleanly(self) -> None:
        if torch_training_available():
            with self.assertRaises(TypeError):
                run_ppo_optimizer_step(
                    PPOTensorBatch(
                        policy_logprobs=None,
                        old_policy_logprobs=None,
                        response_mask=None,
                        advantages=None,
                        returns=None,
                        values=None,
                        old_values=None,
                    ),
                    optimizer=_FakeOptimizer(),
                )
        else:
            with self.assertRaisesRegex(RuntimeError, "optional training dependency"):
                run_ppo_optimizer_step(
                    PPOTensorBatch(
                        policy_logprobs=None,
                        old_policy_logprobs=None,
                        response_mask=None,
                        advantages=None,
                        returns=None,
                        values=None,
                        old_values=None,
                    ),
                    optimizer=_FakeOptimizer(),
                )

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_optimizer_step_backprops_and_updates_policy_and_value_parameters(self) -> None:
        import torch

        policy = torch.nn.Parameter(torch.tensor([[0.0, math.log(1.1)]]))
        values = torch.nn.Parameter(torch.tensor([[0.0, 0.5]]))
        optimizer = torch.optim.SGD([policy, values], lr=0.1)
        batch = PPOTensorBatch(
            policy_logprobs=policy,
            old_policy_logprobs=torch.zeros((1, 2)),
            response_mask=torch.tensor([[0.0, 1.0]]),
            advantages=torch.tensor([[0.0, 1.0]]),
            returns=torch.tensor([[0.0, 1.0]]),
            values=values,
            old_values=torch.zeros((1, 2)),
            entropy=torch.zeros((1, 2)),
        )
        before_policy = policy.detach().clone()
        before_values = values.detach().clone()

        result = run_ppo_optimizer_step(
            batch,
            optimizer=optimizer,
            config=PPOOptimizerStepConfig(
                loss=PPOLossConfig(kl_beta=0.0, value_clip_epsilon=None),
            ),
        )

        self.assertTrue(result.optimizer_step)
        self.assertEqual(result.total_response_tokens, 1)
        self.assertIsNotNone(result.grad_norm)
        self.assertLess(result.backward_loss, 0.0)
        self.assertGreater(result.value_loss, 0.0)
        self.assertFalse(torch.equal(before_policy, policy.detach()))
        self.assertFalse(torch.equal(before_values, values.detach()))

    @unittest.skipUnless(torch_training_available(), "Torch is an optional training dependency")
    def test_optimizer_step_can_accumulate_without_stepping(self) -> None:
        import torch

        policy = torch.nn.Parameter(torch.tensor([[0.0, math.log(1.1)]]))
        values = torch.nn.Parameter(torch.tensor([[0.0, 0.5]]))
        optimizer = torch.optim.SGD([policy, values], lr=0.1)
        batch = PPOTensorBatch(
            policy_logprobs=policy,
            old_policy_logprobs=torch.zeros((1, 2)),
            response_mask=torch.tensor([[0.0, 1.0]]),
            advantages=torch.tensor([[0.0, 1.0]]),
            returns=torch.tensor([[0.0, 1.0]]),
            values=values,
            old_values=torch.zeros((1, 2)),
            entropy=torch.zeros((1, 2)),
        )
        before_policy = policy.detach().clone()

        result = run_ppo_optimizer_step(
            batch,
            optimizer=optimizer,
            config=PPOOptimizerStepConfig(
                loss=PPOLossConfig(kl_beta=0.0, value_clip_epsilon=None),
                gradient_accumulation_steps=2,
                step_optimizer=False,
                max_grad_norm=None,
            ),
        )

        self.assertFalse(result.optimizer_step)
        self.assertIsNone(result.grad_norm)
        self.assertAlmostEqual(result.backward_loss, result.loss / 2.0)
        self.assertFalse(torch.equal(policy.grad, torch.zeros_like(policy.grad)))
        self.assertTrue(torch.equal(before_policy, policy.detach()))


class _FakeOptimizer:
    def zero_grad(self, *args: object, **kwargs: object) -> None:
        return None

    def step(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()
