# Milestone 19: The Training Core Skeleton

This milestone starts Phase 4: the real training core.

It is deliberately not a model-training milestone yet. No weights are updated.
No Transformers model is loaded. Instead, this is the layer that defines what
the real trainer will consume: token masks, logprobs, backend choices, and
runtime config.

That might sound like scaffolding, but for RL on code it is important
scaffolding.

## Why masks come first

A coding prompt has at least two regions:

- the prompt tokens, which describe the task;
- the response tokens, which are the generated program.

The policy-gradient loss should apply to the response tokens, not to the prompt.
That sounds obvious, but it is exactly the kind of detail that can become a bug
if it is not represented directly in the data model.

So I added explicit prompt, response, and attention masks. The tests check that
the masks do not overlap and that padded tokens are excluded. This gives future
GRPO/PPO code a simple contract: compute losses and summaries over the response
mask.

## Logprob records

Real GRPO and PPO need token logprobs. PPO also needs old-policy logprobs for
importance ratios. Both methods need reference-model logprobs for KL tracking.

The new `TokenLogprobs` record stores:

- current policy logprobs;
- optional reference logprobs;
- optional old-policy logprobs;
- optional entropy values.

The summary helper computes response-only means for policy logprob, reference
logprob, KL, importance ratio, and entropy. These are plain Python tuples right
now, not Torch tensors. That is intentional: the contract can be tested without
installing the training stack.

## Backend selection

I added a small backend registry with four entries:

- `smoke`: dependency-free diagnostics, no weight updates;
- `from_scratch`: the first CodeSelf-owned Transformers/PEFT trainer target;
- `trl`: future Hugging Face TRL integration;
- `verl`: future larger-scale RL backend.

The registry can report which optional packages are missing. This does not
validate hardware yet, but it gives the config system a clean place to say
"this experiment wants the from-scratch backend" instead of baking that choice
into scripts.

## Config records

The new training-core config includes:

- model and tokenizer names/revisions;
- dtype, device, LoRA/full-finetune choice, and gradient checkpointing;
- optimizer settings;
- rollout sequence limits and sampling settings;
- algorithm, backend, KL beta, clipping, seed, and max steps.

I also added a debug config for a Qwen2.5-Coder 0.5B-class target. On the local
MacBook M5 Pro with 48GB unified memory, this config is mainly for inspection
and future tiny debug runs. The main paper-grade path should still be portable
to a single CUDA GPU with bf16 LoRA.

## Inspecting the skeleton

There is a small CLI now:

```bash
python3 scripts/inspect_training_core.py \
  --config configs/experiments/training_core_debug.example.json
```

It prints the parsed training-core config and backend availability. For the
debug config, the backend is `smoke`, so it is available without heavy
dependencies. Switching the backend to `from_scratch` would show which training
packages need to be installed.

## What comes next

The next milestone should connect this skeleton to an optional local model
engine: tokenizer encoding, prompt/response truncation, generation output
records, and logprob extraction. After that, the actual GRPO loss has something
stable to stand on.
