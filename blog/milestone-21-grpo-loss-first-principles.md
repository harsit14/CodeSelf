# Milestone 21: GRPO Loss From First Principles

This milestone starts Phase 5: real GRPO.

It still does not train a model. That sounds evasive, but it is the right
sequence. Before attaching Torch tensors, LoRA adapters, and optimizer steps, I
want the project to have a dependency-free implementation of the actual GRPO
objective that can be tested with tiny hand-written examples.

This gives us something important: a reference implementation of the math.

## From rewards to advantages

The smoke trainer already computed group-relative advantages from rollout
records. The new implementation does the training-core version. Given a
`TrainingBatch`, it groups samples by `task_id`, computes the mean and standard
deviation of rewards inside each group, and writes normalized advantages back
onto the corresponding `SequenceTrainingSample` records.

If a group has no reward variance, every sample receives advantage zero. That
case is easy to ignore, but it matters in code RL. Some prompts produce all
wrong answers early in training, or all correct answers on trivial tasks. Those
groups carry no preference signal for GRPO.

## Response-token-only loss

The loss helper consumes the Phase 4 records:

- `SequenceTrainingSample`;
- `TokenLogprobs`;
- prompt/response masks;
- optional reference logprobs;
- old-policy logprobs.

For every response token, it computes the policy ratio between the current
policy and the old policy. Then it applies the PPO-style clipped surrogate used
by GRPO:

```text
min(ratio * advantage, clipped_ratio * advantage)
```

The sample policy loss is the negative mean of that surrogate over response
tokens. Prompt tokens are ignored through the response mask. This is the main
invariant I want protected before tensorization: the model should learn from the
generated code, not from the task prompt.

## KL as a separate term

The helper also supports a reference-model KL penalty when reference logprobs
are present. It reports policy loss, KL loss, total loss, approximate KL, mean
ratio, and clipped-token fraction separately.

Separating these diagnostics now should make later training runs easier to
debug. If reward improves while KL explodes, or if nearly every token is clipped,
the metrics will tell us which part of the objective is misbehaving.

## Why this is still plain Python

This file uses floats, not Torch tensors. That is intentional. On my local
MacBook M5 Pro with 48GB unified memory, these tests run instantly and do not
need any model packages. They give the future Torch implementation a target to
match.

The next step is to tensorize the same objective. After that, the training loop
can connect the model engine, execution rewards, group advantages, and optimizer
step into the first actual GRPO update.
