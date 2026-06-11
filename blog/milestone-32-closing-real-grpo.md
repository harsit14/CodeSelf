# Milestone 32: Closing the Real GRPO Phase

This milestone closes Phase 5.

The Real GRPO phase started as a pile of missing pieces: group-relative
advantages, clipped response-token losses, old-policy ratios, reference KL,
model-forward logprobs, rollouts, checkpoints, and online training. It now has a
complete local path through those pieces.

## What changed in the close-out

Three final gaps were closed.

First, the online trainer can now evaluate on a dev slice after each cycle. If
`eval_tasks` are supplied, the runner writes:

```text
cycle_0001/eval_rollouts.jsonl
cycle_0001/evaluation.json
```

and attaches the evaluation summary to that cycle result.

Second, checkpoints now preserve `save_pretrained()` artifacts when the policy
model exposes that method. That is the hook PEFT/LoRA-style adapters use, and it
means adapter files can sit beside the Torch state files in the same checkpoint
directory.

Third, `ModelEngineCodeGenerator` adapts the training `ModelEngine` surface to
the rollout generator surface. That gives us a clean way to use one shared model
engine for generation, tokenization, and GRPO training.

## Phase 5 is done

The GRPO scaffold now has:

- group-relative advantages;
- response-only clipped policy loss;
- optional reference KL;
- Torch tensor training;
- model-forward logprobs;
- rollout collection and execution rewards;
- rollout-to-training-batch conversion;
- repeated online cycles;
- old-policy snapshot syncing;
- per-cycle dev evaluation;
- state and adapter-style checkpoints.

There is still plenty to build, but it no longer belongs to Real GRPO. The next
major phases are PPO parity, evaluation/dashboarding, self-debug training, and
the final paper-ready package.
