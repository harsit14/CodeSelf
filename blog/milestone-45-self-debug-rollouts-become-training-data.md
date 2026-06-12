# Milestone 45: Self-Debug Rollouts Become Training Data

Phase 8 starts by turning the existing self-debug agent loop into a normal
rollout source.

CodeSelf already had an agentic smoke loop: generate code, run public tests,
observe the failure, apply a small deterministic repair, and submit a final
answer. That was useful as a trace artifact, but it lived next to the rollout
pipeline rather than inside it. For RL training, that distinction matters. GRPO
and PPO expect `RolloutRecord` objects, not bespoke agent traces.

This milestone adds the bridge.

## What landed

The new module is `self_debug_rollouts.py`.

It adds:

- `SelfDebugRolloutConfig`;
- `SelfDebugRolloutResult`;
- `generate_self_debug_rollouts`;
- `rollout_record_from_trace`.

The key idea is simple: the trace remains the audit trail, but the final revised
program becomes a standard rollout response. The resulting record still has the
normal prompt, parsed code, execution payload, reward payload, backend name, and
model name. It also carries self-debug metadata: revision count, tool-call
count, public-test pass status, final pass status, initial parse status, and the
active rollout mode.

`scripts/run_rollouts.py` now supports:

```bash
python3 scripts/run_rollouts.py \
  --rollout-mode self_debug \
  --trace-output outputs/reports/self_debug_traces.jsonl \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/self_debug.jsonl
```

Direct single-shot rollouts remain the default.

## Reward discounting

The first reward rule is deliberately conservative. The final attempt is still
the attempt that gets rewarded. If `revision_reward_discount` is less than
`1.0`, positive final rewards are multiplied by:

```text
discount ** revision_count
```

That makes a one-shot correct answer worth slightly more than an answer that
needed public-test feedback, while still allowing revised solutions to train
the model. Non-positive rewards are left unchanged so failed attempts do not
become less negative just because the agent spent more rounds.

The undiscounted final reward and discount factor are preserved in reward
metrics, which should make later analysis less ambiguous.

## Ablation configs

I added two matching config templates:

- `rollout_single_shot_ablation.example.yaml`;
- `rollout_self_debug_ablation.example.yaml`.

They are intentionally close to each other. The goal is to make single-shot
versus self-debug comparisons a config difference, not a separate code path.

## What I tested

The new tests check the Python API and the CLI. They verify that a failing
initial solution can be repaired into a passing final rollout, that the final
reward can be discounted by revision count, that the rollout metadata records
the self-debug path, and that trace JSONL is written alongside rollout JSONL.

I also ran the GRPO and PPO rollout-cycle tests because the whole point of this
slice is compatibility with training consumers. The self-debug output is just a
normal rollout record, which means the existing batch builders do not need a
special case.

## What is next

The repair policy is still smoke-test machinery. The next real research slice
should make revision generation model-driven: construct a feedback prompt from
public-test observations, sample a revised answer, and log the prompt/response
pair for training. After that, online GRPO and PPO should get a config switch
that collects self-debug rollouts natively inside each training cycle.
