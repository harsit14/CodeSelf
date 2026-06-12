# Milestone 47: Self-Debug Enters Online Training Cycles

Phase 8 now reaches the online training loop.

The previous self-debug work made revised attempts usable as normal
`RolloutRecord` objects. That was enough for pre-generated rollout files, but
not enough for online GRPO and PPO. In an online RL run, the trainer should be
able to collect the rollout, execute feedback, revise, score, batch, and update
inside the same cycle.

This milestone adds that native path.

## What landed

I added `SelfDebugCollectionConfig` for training-cycle self-debug settings:

- max revisions;
- revision strategy;
- revision prompt template;
- revision reward discount.

Both `GRPORolloutTrainingCycleConfig` and `PPORolloutTrainingCycleConfig` now
have `rollout_mode`. The default is still `direct`. Setting it to `self_debug`
makes the cycle call `generate_self_debug_rollouts()` before building the GRPO
or PPO training batch.

The nice part is what did not change. The batch builders still consume ordinary
rollout records. The GRPO/PPO training objectives still see prompts, responses,
rewards, and metadata in the same shape as before.

## Artifacts

Self-debug cycles can now write:

```text
rollouts.jsonl
self_debug_traces.jsonl
metrics.jsonl
checkpoint.json
state/
```

The online GRPO and PPO runners write `self_debug_traces.jsonl` into each
`cycle_*` directory whenever that cycle uses self-debug collection. This keeps
the training-ready rollout file compact while preserving the full public-test
feedback and revision trace beside it.

## Why this matters

For this project, self-debugging is not just an evaluation trick. It is a data
collection mode. If a model learns from execution feedback, the training loop
needs to know whether the sampled answer was single-shot or revised after a
tool observation.

Now that distinction is explicit in rollout metadata:

```text
rollout_mode = direct | self_debug
revision_count = ...
revision_strategy = ...
```

That should make later ablations much cleaner.

## What I tested

The new tests cover both algorithms twice: once at the one-cycle layer and once
at the online layer. They verify self-debug rollout collection, trace artifact
writing, discounted final rewards, rollout metadata, and continued compatibility
with the existing GRPO/PPO batch builders.

The full suite still runs with the direct rollout path as the default, so this
feature is opt-in and should not disturb existing smoke experiments.

## What is next

The next practical missing piece is a real launch surface for online training
configs. Right now the typed Python records can express self-debug online
cycles, but the repository still needs a JSON/YAML config loader and script for
running GRPO/PPO online experiments end to end from the command line.
