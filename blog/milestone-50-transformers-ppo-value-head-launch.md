# Milestone 50: PPO Gets a Transformers Value Head

This milestone closes the most important gap left by the first online launcher.

The launcher could already run GRPO with a local Transformers causal LM. PPO was
different. PPO needs value estimates, and a plain causal-LM head only gives
token logits. In the previous milestone I made that limitation explicit: PPO
with `model.backend=transformers` failed early instead of stumbling into a
half-assembled run.

Now the launcher has the missing value-head path.

## What landed

I added:

```text
src/codeself/training/ppo_value_head.py
```

The core class is:

```text
CausalLMWithValueHead
```

It wraps a causal LM, requests hidden states during the forward pass, and
applies a small linear head to produce token-level values. The PPO tensor bridge
already knew how to consume outputs shaped like:

```text
logits: [batch, tokens, vocab]
values: [batch, tokens]
```

So the cleanest fix was to make a Transformers policy model speak that existing
contract.

## How Launch Works Now

For PPO with:

```text
model.backend: transformers
```

the launcher now:

- loads the local Transformers causal LM;
- wraps it with `CausalLMWithValueHead`;
- keeps rollout generation tied to the same underlying policy model;
- passes the wrapped policy/value model into the online PPO runner;
- optionally assembles old-policy and old-value snapshots when configured.

This matters because online RL can go wrong in subtle ways if rollout sampling
and optimization point at different model objects. Here they share the same base
policy weights. The wrapper adds values; it does not fork the policy.

## The New Template

I added:

```text
configs/experiments/ppo_online_transformers_launch.example.yaml
```

Like the GRPO launch template, it is conservative for a MacBook Pro with 48 GB
unified memory:

```text
device: mps
training.device: mps
dtype: fp16
local_files_only: true
group_size: 2
samples_per_task: 2
```

This is not a promise that every model will fit. It is a sane starting point for
small cached models.

## What I Tested

I did not want this test to depend on downloaded Hugging Face weights, so I used
a fake local Transformers engine. The test still exercises the real launcher,
the real value-head wrapper, and the real PPO online training path.

The new coverage checks:

- the value-head wrapper returns logits and values with the expected shapes;
- PPO with `model.backend=transformers` runs through the single-config launcher;
- the resulting PPO training result records `CausalLMWithValueHead` as the
  policy model class;
- the new PPO Transformers launch example parses through the online config
  builder.

## What Is Still Not Done

This is full fine-tuning with an integrated value head. It is not LoRA, and it
is not a separately pretrained critic. Those are real future improvements, but
they are narrower than the previous blocker. The important change is that PPO
now has an honest local Transformers launch path.

That moves the scaffold one step closer to being a research tool instead of a
collection of individually correct components.
