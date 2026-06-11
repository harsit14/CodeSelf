# Milestone 28: Turning Rollouts into GRPO Training Batches

This milestone connects two pieces that were previously sitting next to each
other: execution rollouts and the GRPO trainer.

Before this step, CodeSelf could generate, execute, and score solutions. It
could also train a tiny Torch causal-LM-like model with a GRPO objective. But
there was still a missing adapter between the rollout JSONL world and the
`TrainingBatch` world. The trainer understood token IDs, response masks,
rewards, and advantages. The rollout system produced prompts, completions,
parsed code, execution results, and reward dictionaries.

The new bridge is `build_grpo_training_batch_from_rollouts`.

## The default: train on what the model actually said

The default response source is the raw completion. That is intentional. In
policy-gradient training, the update should usually correspond to the exact
tokens sampled from the policy. If the model emitted Markdown fences, prose, or
some extra explanation, those tokens are part of the sampled trajectory.

There is also a `parsed_code` option. That will be useful for experiments where
we explicitly want to train only on extracted Python code. I do not want that to
be the accidental default, though, because it changes the trajectory being
optimized.

## Skips are first-class diagnostics

The bridge can skip records that cannot produce a useful GRPO sample:

- failed parses, when configured to exclude them;
- empty responses;
- responses that tokenize to zero response tokens;
- invalid rewards.

If every record is skipped, the builder raises an error. That is the right kind
of annoying. A rollout pipeline that produces no trainable samples should fail
loudly during research, not quietly produce an empty-looking improvement.

## Advantages happen at the boundary

After tokenization, the bridge immediately calls the existing
group-relative-advantage code. That means the result is ready for the tensor
batch builder and model-aware trainer.

The current chain now looks like:

```text
RolloutRecord -> TrainingBatch with advantages -> model forward
  -> GRPO tensor batch -> optimizer step -> metrics/checkpoint
```

This is still not the full online RL loop. The next missing piece is a live
collector that samples multiple completions per prompt from the current policy,
executes them, scores them, turns them into a batch, and optimizes immediately.
But this milestone removes the awkward handoff between "we have executions" and
"we have a trainable GRPO batch."
