# Milestone 29: A One-Cycle GRPO Rollout Trainer

This milestone is the first time the Phase 5 pieces behave like a small RL
cycle instead of separate tools.

The new entry point is `run_grpo_rollout_training_cycle`. It does four things:

- collects multiple rollouts per task;
- executes and scores them through the existing rollout path;
- converts the resulting `RolloutRecord` objects into a GRPO `TrainingBatch`;
- runs the model-aware GRPO trainer on that batch.

That sounds simple, but it is an important boundary. Earlier milestones proved
the trainer in isolation and then built an adapter from rollouts to training
batches. This one composes them into a single call.

## The shape of one cycle

The path now looks like this:

```text
tasks + generator -> RolloutRecord[] -> TrainingBatch
  -> model-forward GRPO tensors -> optimizer step -> metrics/checkpoint
```

The cycle can also write the rollout JSONL before training. I like that because
it preserves the sampled trajectories that produced the update. For execution
feedback RL, that audit trail is not a luxury; it is how we debug weird reward
curves later.

## KL is optional at the cycle boundary

The cycle default uses `kl_beta=0.0`. That makes it runnable without a reference
model, which is useful for local development and tiny integration tests.

The lower-level trainer still supports a reference model and nonzero KL. The
cycle just does not force that on the first path.

## What this is not yet

This is not a many-step online trainer. There is no policy snapshot cadence, no
dev-set evaluation loop, and no automatic handoff between a Transformers policy
object and the generator. The current generator and trainable model can still be
different objects.

But the scaffold now has a real spine:

```text
collect -> execute -> reward -> batch -> optimize
```

That is the path the next milestones can turn into a repeated online training
run.
