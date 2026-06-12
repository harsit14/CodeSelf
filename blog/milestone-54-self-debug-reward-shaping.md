# Milestone 54: Self-Debug Reward Shaping Becomes Explicit

This milestone cleans up the last Phase 8 caveat.

The self-debug rollout path already had `revision_reward_discount`, but the
rule was hardcoded: if the final reward was positive, multiply it by
`discount ** revision_count`; otherwise leave it alone.

That was fine as a first pass. It also hid an experimental assumption.

Today I made that assumption configurable.

## The New Fields

Self-debug configs now support:

```text
revision_reward_discount_mode
revision_reward_step_penalty
```

The default mode is still:

```text
positive_only
```

That preserves old behavior. Passing solutions get discounted when they needed
extra revisions. Failed or non-positive solutions stay unchanged unless the
experiment asks for another policy.

There are two other modes:

```text
all
none
```

`all` applies the discount factor to every final reward, positive or negative.
`none` leaves the final reward untouched.

The step penalty is separate. It subtracts a fixed cost per revision after the
discount is applied. That gives us a clean way to charge extra tool-use or
repair attempts even when the final answer still fails.

## Why This Matters

Reward shaping is one of those places where an implementation detail can become
an accidental research claim.

If self-debug improves pass rate, we need to know what the objective actually
encouraged. Did it reward final correctness only? Did it penalize needing a
repair? Did it penalize failed repairs too? Did it discount negative outcomes?

Those should be config choices, not archaeology.

## Traceability

Rollout reward metrics now preserve:

```text
self_debug_undiscounted_final_reward
self_debug_discounted_reward_before_penalty
self_debug_revision_count
self_debug_revision_reward_discount
self_debug_reward_discount_mode
self_debug_discount_factor
self_debug_revision_step_penalty
self_debug_step_penalty_total
self_debug_shaped_reward
```

The final scalar reward is still easy to consume, but the artifact explains how
it got there.

## Where It Flows

The new fields pass through:

- `SelfDebugRolloutConfig`;
- `SelfDebugCollectionConfig`;
- GRPO and PPO self-debug collection cycles;
- online training config loading;
- `scripts/run_rollouts.py`;
- online config inspection;
- self-debug rollout and online training templates.

The defaults keep existing configs compatible, while newer ablations can choose
their shaping policy explicitly.

## What I Tested

The tests now cover:

- the old positive-only behavior with an added step penalty;
- `all` mode on a negative final reward;
- CLI propagation through `run_rollouts.py`;
- online config parsing of the new shaping fields.

At this point Phase 8 has crossed a useful threshold. The scaffold no longer
just says "self-debug training mode" on paper; it can collect traces, run
model-generated revisions, train online GRPO/PPO loops, launch local
Transformers policies with LoRA, use a separate PPO critic, and make reward
shaping auditable.
