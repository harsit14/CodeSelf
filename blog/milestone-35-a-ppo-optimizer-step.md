# Milestone 35: A PPO Optimizer Step

PPO now has a real optimizer-step helper.

The previous milestone made the PPO objective differentiable with Torch. This
one wraps that objective in the training action we actually need: backpropagate,
clip gradients, optionally step the optimizer, and return useful metrics.

## What landed

The new module is `ppo_step.py`.

It adds:

- `PPOOptimizerStepConfig`;
- `PPOOptimizerStepResult`;
- `run_ppo_optimizer_step`.

The helper mirrors the GRPO optimizer-step shape. It supports gradient
accumulation, max-grad-norm clipping, optional zero-grad, optional optimizer
stepping, and detached diagnostics for policy loss, value loss, entropy loss,
KL loss, ratios, returns, and clipping fractions.

## Why this matters

PPO has two trainable surfaces: the policy and the value head. The tests now
verify gradients move both policy logprobs and value predictions. There is also
a no-step accumulation test, which is important for the later training loop.

The next PPO gap is model wiring: a causal-LM policy with a value head that can
produce the tensors this optimizer step consumes.
