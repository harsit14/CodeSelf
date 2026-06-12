# Milestone 40: Closing PPO Parity

PPO now has a repeated online training loop.

The previous milestone gave PPO a single collect-to-optimize cycle. This one
turns that into the same kind of repeated online path that GRPO already had:
collect rollouts, score them, train, preserve optimizer state, optionally sync
snapshots, and evaluate after each cycle.

## What landed

The new module is `ppo_online.py`.

It adds:

- `PPOOnlineEvaluationConfig`;
- `PPOOnlineTrainingConfig`;
- `PPOOnlineEvaluationResult`;
- `PPOOnlineTrainingStep`;
- `PPOOnlineTrainingResult`;
- `run_ppo_online_training`.

The runner keeps a single optimizer across cycles when one is not supplied.
That detail matters: rebuilding AdamW every cycle would make the online loop
look correct from the outside while quietly discarding optimizer state.

## PPO-specific snapshot sync

GRPO only needs an old-policy snapshot. PPO needs two stale baselines:

- old policy log probabilities for the clipped policy ratio;
- old value predictions for GAE and clipped value loss.

The new config therefore has separate switches for old-policy and old-value
syncing. Each cycle records whether those snapshots were synced, so the run
metadata can explain the baseline used for that update.

## Evaluation parity

PPO now shares the same dev-evaluation cadence shape as GRPO. If eval tasks are
provided, the runner can generate evaluation rollouts after each training cycle,
write `eval_rollouts.jsonl`, write `evaluation.json`, and attach the summary to
the cycle result.

This is not the final dashboard or statistical reporting layer. It is the
training-side hook that makes those later plots and tests possible.

## What Phase 6 means now

At this point PPO has:

- scalar reference objective math;
- Torch tensor objective math;
- optimizer steps;
- causal-LM/value-head tensor wiring;
- a tensor training loop;
- model-aware training;
- rollout-to-batch conversion;
- one-cycle rollout training;
- repeated online training;
- old-policy and old-value snapshot syncing;
- dev evaluation cadence.

That is enough to call the PPO baseline scaffold complete. The next phase can
move out of algorithm plumbing and into evaluation, statistics, and dashboards.
