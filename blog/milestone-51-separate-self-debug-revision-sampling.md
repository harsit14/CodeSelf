# Milestone 51: Separating Initial and Revision Sampling

This milestone cleans up an important experimental confound in the self-debug
path.

Until now, the first answer and the model-generated revision shared the same
sampling settings. That was convenient, but it made a bad assumption: producing
an initial solution and repairing a solution after public-test feedback are not
the same generation problem.

The initial attempt may benefit from exploration. The revision pass often wants
to be shorter, colder, and more literal.

## What Landed

I added three optional fields:

```text
revision_max_new_tokens
revision_temperature
revision_top_p
```

They now flow through:

- `AgentLoopConfig`;
- `SelfDebugRolloutConfig`;
- `SelfDebugCollectionConfig`;
- `generate_self_debug_rollouts()`;
- GRPO/PPO self-debug training cycles;
- online config loading;
- rollout and agentic CLI flags.

The defaults are backward-compatible. If a revision field is not set, it
inherits the initial generation setting.

## Traceability

The important part is not only that the settings exist. It is that traces and
rollout records now preserve the resolved revision request:

```text
revision_max_new_tokens
revision_temperature
revision_top_p
revision_request_max_new_tokens
revision_request_temperature
revision_request_top_p
```

That makes later analysis less hand-wavy. If a self-debug run improves pass
rate, the artifact can show whether the revision was sampled with the same
temperature as the initial attempt or with a deliberately colder repair policy.

## The Example Config

I updated:

```text
configs/experiments/rollout_self_debug_model_revision.example.yaml
```

The example now keeps the initial rollout exploratory:

```text
temperature: 0.8
top_p: 0.95
```

while making model revisions more constrained:

```text
revision_max_new_tokens: 384
revision_temperature: 0.2
revision_top_p: 0.9
```

This is the kind of small experimental distinction that becomes very annoying
to reconstruct after the fact if it is not represented in the config.

## What I Tested

The tests now check:

- model revision requests actually use the revision-specific token and sampling
  settings;
- self-debug rollout records preserve resolved revision sampling metadata;
- `run_rollouts.py` propagates the new CLI flags into traces and rollout
  metadata;
- online GRPO/PPO config builders parse the new fields.

This closes one of the last Phase 8 risks. The self-debug setup is now much
better positioned for clean ablations: initial attempt policy, revision policy,
and reward discounting are separate enough to study without guessing.
