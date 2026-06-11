# Milestone 36: PPO Gets a Value-Head Bridge

This milestone gives PPO the model-facing bridge it was missing.

Up to this point, PPO could compute a reference scalar objective, a Torch tensor
objective, and an optimizer step. That was enough to test the math, but not
enough to point a causal language model at the objective. PPO needs both policy
log probabilities and value predictions, so the model wiring has one extra
surface compared with GRPO.

## What landed

The new module is `ppo_model.py`.

It adds:

- `PaddedPPOTrainingTensors`;
- `pad_ppo_training_batch_tensors`;
- `gather_causal_lm_token_entropy`;
- `build_ppo_tensor_batch_from_model`.

The builder runs a policy model over padded token batches, gathers next-token
log probabilities, extracts current value predictions, computes categorical
entropy, and prepares the padded tensors consumed by the PPO objective.

It also supports optional old-policy, old-value, and reference models. Those
baselines are deliberately detached. The PPO update should move the current
policy and current value head, not the rollout snapshot or reference model.

## The small trap

The easy mistake is to let the value target path become differentiable. PPO
advantages and returns should be fixed targets for the optimizer step. The
current value predictions should receive gradients, but the old values used for
GAE and value clipping should not.

The tests now cover that boundary directly. A toy causal-LM with a value table
updates both logits and values through `run_ppo_optimizer_step`, while old
policy logprobs, old values, and reference logprobs stay detached.

## Why this matters

This is the first PPO milestone that looks like the real training surface:

1. tokenize prompt and completion;
2. run the policy/value model;
3. compute old-policy and old-value baselines;
4. build a `PPOTensorBatch`;
5. backpropagate the clipped PPO objective.

The remaining PPO work is now mostly orchestration: a training loop, a
model-aware trainer, rollout-cycle integration, repeated online cycles, and dev
evaluation parity with GRPO.
