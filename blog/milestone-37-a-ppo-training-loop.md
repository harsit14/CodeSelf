# Milestone 37: A PPO Training Loop

PPO now has the same minimal training-loop layer that GRPO got earlier in the
overhaul.

The previous PPO milestones built the pieces separately: scalar objective,
Torch objective, optimizer step, and model/value-head tensor bridge. This
milestone is about orchestration. Once tensor batches exist, the trainer still
needs to decide when to zero gradients, how to accumulate microbatches, when to
step the optimizer, and whether to step a scheduler.

## What landed

The new module is `ppo_loop.py`.

It adds:

- `PPOTrainingLoopConfig`;
- `PPOTrainingLoopStep`;
- `PPOTrainingLoopResult`;
- `run_ppo_training_loop`.

The shape mirrors the GRPO loop. It accepts an iterable of `TrainingBatch`
objects and a tensor-batch builder. That builder can be the simple tensor
builder from tests today or the model-facing PPO builder from the previous
milestone tomorrow.

## Why this layer exists

It is tempting to hide gradient accumulation inside the optimizer step, but the
loop is the level that actually knows about microbatches. The loop can turn a
requested accumulation size into the effective size for each group, including
the final partial group. That matters for small local debug runs where there may
not be an exact multiple of the accumulation window.

The loop also records enough diagnostics to make later trainer artifacts useful:
microbatch count, optimizer-step count, policy loss, value loss, entropy loss,
KL loss, response-token totals, and per-step metadata.

## Tests as a contract

The new tests use tiny Torch parameters for policy log probabilities and value
predictions. They verify that:

- accumulation delays optimizer stepping until the end of a group;
- the final partial group still steps correctly;
- schedulers step only on optimizer steps;
- `max_batches` trims input batches;
- both policy and value parameters update.

That gives PPO a real loop surface without yet taking on rollout generation,
checkpointing, or online training. Those are the next parity layers.
