# Milestone 33: Starting PPO with the Objective

Phase 6 starts with PPO's math before its trainer.

The new module is `ppo_loss.py`. It is deliberately dependency-free, following
the same pattern that worked well for GRPO: make the scalar reference objective
easy to test before moving it into Torch tensors.

## What landed

The first PPO slice adds:

- GAE value targets over response tokens;
- sparse final rewards for coding-task rollouts;
- clipped policy loss;
- clipped value loss;
- entropy bonus;
- optional reference KL;
- advantage normalization;
- per-sample and batch-level diagnostics.

The value API separates current values from rollout-time old values. GAE uses
the old values, while the value loss compares current values against the
returns and clips around the old values.

That distinction matters. PPO is very easy to make look plausible while quietly
using the wrong baseline.

## Why this comes before Torch

The next PPO milestones will need tensor batches, a value head, optimizer
steps, and then an online loop comparable to GRPO. Those pieces are much easier
to debug when the reference math is already nailed down.

For now, Phase 6 has a concrete first brick: a testable PPO objective that
matches the response-token training surface used by the rest of CodeSelf.
