# Milestone 53: A Separate Critic for PPO

This milestone closes the last big launcher architecture gap for PPO.

Until now, the Transformers PPO path used an integrated value head. That was a
good first implementation: load one causal LM, wrap it with
`CausalLMWithValueHead`, and train policy logits and token values together.

But PPO experiments often want another option: a distinct critic/value model.
That critic may start from the same base model, a different checkpoint, or a
previously trained value checkpoint.

Today I added that path to the single-config launcher.

## What Changed

The new config switch is:

```text
model.value_model.enabled: true
```

When this is off, PPO keeps the old integrated layout:

```text
policy causal LM -> PPO value-head wrapper
```

When it is on, PPO now assembles:

```text
policy causal LM
separate critic causal LM -> PPO value-head wrapper
```

That means rollout generation still uses the policy model, while value
prediction comes from a distinct trainable model passed into the PPO trainer as
`value_model`.

## Why This Matters

This is one of those engineering details that changes the experiment without
looking very dramatic in the config.

With an integrated head, the policy and critic share the same base parameters.
That is cheaper and simpler.

With a separate critic, the optimizer can update a different value backbone.
That costs more memory, but it lets us run cleaner ablations around value
learning, policy/value interference, and warm-started critics.

On a local MacBook with 48GB unified memory, the integrated path remains the
friendly default. The separate critic path is now available when the model size
and batch shape make it reasonable.

## Checkpoint Loading

I also added two loading hooks to `CausalLMWithValueHead`:

```text
load_state_path(...)
load_value_head_path(...)
```

The launcher maps those to:

```text
model.value_model.state_path
model.value_model.value_head_state_path
```

The first loads the full wrapper state. The second loads only the value-head
weights, which matches the `value_head.pt` artifact written by the existing
`save_pretrained()` checkpoint path.

## Snapshot Syncing

Old-model syncing needed a little care.

In integrated-head mode, the old policy can double as the old value model,
because it has both logits and values.

In separate-critic mode, that would be wrong. The old policy should mirror the
bare policy architecture, and the old value model should mirror the critic
architecture.

The launcher now follows that split.

## The New Template

I added:

```text
configs/experiments/ppo_online_transformers_separate_value.example.yaml
```

It keeps the LoRA policy path from the previous milestone and adds a separate
LoRA value model. The template is deliberately explicit, even where values
could be inherited, because this is an ablation config and future-me deserves
not to guess which model was doing value prediction.

## What I Tested

The tests cover:

- PPO Transformers launch with an integrated value head;
- PPO Transformers launch with a separate value model;
- old-policy and old-value syncing in the separate critic layout;
- full value-model state loading;
- value-head-only checkpoint loading.

The remaining Phase 8 caveat is now much smaller: the self-debug reward
discount is still intentionally simple. The model launcher side is finally in
shape for real local ablations.
