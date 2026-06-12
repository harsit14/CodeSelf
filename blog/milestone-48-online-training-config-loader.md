# Milestone 48: Making Online Training Configs Inspectable

The next Phase 8 step was not a new optimizer trick. It was a launch-surface
step, which sounds less glamorous until you remember how easy it is to waste an
afternoon running the wrong RL experiment.

After the previous milestone, CodeSelf could express self-debug collection
inside online GRPO and PPO cycles. The missing piece was that those settings
mostly lived as Python dataclasses. That is fine for tests, but it is not how I
want to run ablations. A real experiment should start from a JSON or YAML file
that can be checked into git, inspected, copied, and compared.

This milestone adds that bridge.

## What landed

I added a new online config builder module with three entry points:

```text
build_grpo_online_training_config()
build_ppo_online_training_config()
build_online_training_config()
```

The builders take the nested experiment mappings already used elsewhere in the
project and turn them into the typed online training records:

```text
GRPOOnlineTrainingConfig
PPOOnlineTrainingConfig
```

They cover the pieces that matter for launch shape:

- cycle count and seed stride;
- rollout size and sampling settings;
- direct versus self-debug collection;
- self-debug revision policy and reward discounting;
- batch conversion settings;
- optimizer and loss settings;
- optional evaluation settings.

I also added `resolve_online_algorithm()` so scripts can consistently decide
whether a config is GRPO or PPO.

## The Small CLI

The new script is:

```text
scripts/inspect_online_training_config.py
```

It does not launch a model yet. That is intentional. Instead, it loads a
JSON/YAML config, builds the typed online config, prints a concise summary, and
can write a normalized JSON artifact.

For example, the output includes:

```text
algorithm: grpo
cycles: 2
rollout_mode: self_debug
samples_per_task: 4
self_debug_revision_strategy: rule_based
evaluation_enabled: true
```

That gives me a cheap sanity check before spending GPU time. In RL scaffolding,
this kind of boring validation is a real piece of research infrastructure.

## Example Configs

I added two templates:

```text
configs/experiments/grpo_online_self_debug.example.yaml
configs/experiments/ppo_online_self_debug.example.yaml
```

Both describe a two-cycle self-debug online run with evaluation enabled. They
are not meant to pretend the full model-loading launcher is done. They are
meant to make the intended launch contract concrete.

## Why This Matters

The project is now juggling several axes:

- GRPO versus PPO;
- direct rollouts versus self-debug rollouts;
- rollout generation settings;
- training objective settings;
- online cycle state, including old-policy snapshots;
- optional evaluation after each cycle.

Keeping that only in Python constructors would make the system hard to audit.
The config loader gives every experiment a portable paper trail. If a run looks
good or bad later, I can inspect the exact rollout mode, KL setting, seed stride,
and revision policy that produced it.

That is the part that feels most like grad school: half the work is making sure
future-me can reconstruct what past-me actually did.

## What I Tested

The new unit tests cover:

- GRPO config building from a nested mapping;
- PPO config building from a nested mapping;
- loading both example YAML configs into typed online configs;
- running the inspector CLI and checking its normalized JSON output.

This closes the config-loader hole that was sitting in Phase 8's risk list.
The next remaining engineering jump is a full launcher that assembles tasks,
generators, tokenizers, models, optimizers, and artifact paths from one config
file. That will be more invasive, so I am glad this validation layer now exists
first.
