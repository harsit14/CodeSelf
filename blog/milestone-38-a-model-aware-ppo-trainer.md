# Milestone 38: A Model-Aware PPO Trainer

PPO now has a model-aware trainer.

The previous milestone added the tensor training loop. That loop knew how to
accumulate gradients and step an optimizer, but it did not know anything about
models, value heads, checkpoints, or metric files. This milestone adds that
outer layer.

## What landed

The new module is `ppo_trainer.py`.

It adds:

- `PPOModelTrainingConfig`;
- `PPOCheckpointArtifact`;
- `PPOModelTrainingResult`;
- `PPOValueEstimateProvider`;
- `run_ppo_model_training`.

The trainer takes `TrainingBatch` objects, runs the model-facing PPO tensor
builder, sends those batches through `run_ppo_training_loop`, and returns a
checkpoint-ready result object.

## Why PPO needs a little more surface area

GRPO only needs a policy model and, optionally, old-policy and reference
models. PPO also needs value predictions. The trainer supports both styles I
expect to use while experimenting:

- a policy model with an integrated value head;
- a separate value model paired with a logits-only policy model.

It also accepts old-policy, old-value, and reference models. Those snapshots
are put in eval mode, while the trainable policy and value surfaces are put in
train mode.

## Artifacts

The trainer can now write:

- JSONL metrics, one row per microbatch;
- a checkpoint manifest;
- Torch state files for the policy, optional value model, optimizer, and
  scheduler;
- adapter-style `save_pretrained()` artifacts when the model exposes that API.

The checkpoint manifest includes file sizes and SHA-256 hashes so future runs
can distinguish "I wrote a manifest" from "I preserved the actual training
state."

## What this unlocks

PPO now has the same model-level training surface that GRPO had before rollout
orchestration was added. The next PPO parity gap is the collect-to-optimize
cycle: generate rollouts, turn them into PPO training batches with value
baselines, run this trainer, and preserve the resulting artifacts.
