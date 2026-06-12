# Milestone 39: A PPO Rollout Training Cycle

PPO now has its first collect-to-optimize cycle.

The previous milestone made PPO model-aware: given a `TrainingBatch`, it could
run a policy/value model, build tensor batches, optimize, and write checkpoint
artifacts. This milestone connects that trainer to execution-feedback rollouts.

## What landed

Two new modules landed.

`ppo_rollouts.py` adds:

- `PPORolloutBatchConfig`;
- `PPORolloutBatchResult`;
- `build_ppo_training_batch_from_rollouts`.

`ppo_cycle.py` adds:

- `PPORolloutTrainingCycleConfig`;
- `PPORolloutTrainingCycleResult`;
- `run_ppo_rollout_training_cycle`.

The rollout batch bridge looks a lot like the GRPO bridge, but it deliberately
does not assign group-relative advantages. PPO keeps raw scalar rewards in the
samples, then computes GAE from value baselines in the model-facing PPO path.

## Why this slice matters

This is the first PPO path that starts with generated code and ends with a
model update:

1. render prompts;
2. sample completions;
3. parse and execute them;
4. score the execution result;
5. tokenize prompt/completion pairs;
6. build a PPO training batch;
7. run the model-aware PPO trainer;
8. write rollout, metric, checkpoint, and state artifacts.

That is the shape a real execution-feedback RL experiment needs. It is still a
tiny one-cycle version, but the boundary is now honest.

## A GRPO difference worth preserving

GRPO needs multiple samples per prompt because the advantage signal is
group-relative. PPO can still benefit from multiple samples, but the batch
builder does not require group normalization. That distinction is now visible in
the code: `build_grpo_training_batch_from_rollouts` assigns advantages,
`build_ppo_training_batch_from_rollouts` does not.

## Tests

The dependency-free tests verify rollout conversion, parse-failure skipping,
metadata carry-through, token limits, and config validation.

The Torch cycle test runs the whole path with a tiny integrated policy/value
model. It confirms that rollouts are written, PPO metrics are written, checkpoint
state artifacts exist, and both policy logits and value predictions update.

The next PPO parity layer is repeated online training: preserve optimizer state
across cycles, optionally sync old-policy and old-value snapshots, and add dev
evaluation cadence like GRPO.
