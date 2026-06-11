# Milestone 30: Repeating the GRPO Cycle

The previous milestone gave CodeSelf one complete GRPO cycle:

```text
collect -> execute -> reward -> batch -> optimize
```

This milestone repeats that cycle.

The new entry point is `run_grpo_online_training`. It runs multiple
rollout-training cycles, increments the seed between cycles, keeps one optimizer
alive across the run, and writes cycle-scoped artifacts when an artifact
directory is provided.

## Why this matters

A single update is a great integration test, but it is not really training. The
first online loop needs to answer basic research-engineering questions:

- did every cycle collect the intended number of rollouts?
- did each cycle produce trainable records?
- how many optimizer steps actually happened?
- where are the rollouts and checkpoint files for cycle N?

The new result object answers those directly. It reports total rollouts,
records used, total optimizer steps, mean reward, and a per-cycle result.

## Artifact layout

When `artifact_dir` is set, each cycle writes to its own directory:

```text
artifact_dir/
  cycle_0001/
    rollouts.jsonl
    metrics.jsonl
    checkpoint.json
    state/
  cycle_0002/
    ...
```

This is intentionally boring. Boring artifact layouts are easier to audit after
a run fails halfway through.

## Still missing

This is not yet the final online trainer. The policy model and generator can
still be separate objects. The old-policy model is not snapshotted between
cycles, and there is no dev-set evaluation cadence yet.

But the scaffold now has a repeated training spine. We can run more than one
collect-and-update cycle and inspect the trail it leaves behind.
