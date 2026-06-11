# Milestone 31: Old-Policy Snapshots for Online GRPO

This milestone makes the repeated GRPO loop less hand-wavy about the "old
policy" in policy-gradient training.

In GRPO, the clipped objective compares the current policy against the policy
that produced the samples. The scaffold already knew how to accept an
`old_policy_model`, but the online loop did not manage that model across cycles.
If the caller passed one, it stayed whatever it was at construction time.

Now the online runner can sync it before every cycle.

## What changed

`GRPOOnlineTrainingConfig` has a new default-on flag:

```text
sync_old_policy_before_cycle = true
```

When an old-policy model is supplied, the runner copies:

```text
policy_model.state_dict() -> old_policy_model.load_state_dict(...)
```

before collecting and optimizing that cycle.

Each `GRPOOnlineTrainingStep` also records `old_policy_synced`, so serialized
results show whether a cycle actually used a fresh snapshot.

## Why this is only one layer

This does not magically make the generator and policy model the same object.
That wiring still belongs to the Transformers integration layer.

But it does mean the model-training side now has a sane snapshot cadence:

```text
sync old policy -> collect rollouts -> build batch -> optimize policy
```

For a research scaffold, that is the difference between "we pass an optional
argument" and "we can explain which policy the ratio baseline came from."

## Roadmap note

After Real GRPO is wrapped up, there are four major named phases left in the
overhaul plan: PPO baseline, evaluation/dashboard, self-debug training mode,
and final documentation/paper packaging.
