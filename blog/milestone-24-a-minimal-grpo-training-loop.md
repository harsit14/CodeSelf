# Milestone 24: A Minimal GRPO Training Loop

This milestone adds the first small loop around the GRPO optimizer step.

The last two milestones made the objective differentiable and added a narrow
optimizer-step helper. That gave the project the parts of a training step, but
not the control flow around microbatches. This milestone adds that control flow.

## Running with active Torch tests

I created a repo-local Python 3.12 virtual environment and installed PyTorch
there. The system `python3` points at Python 3.14, which did not have Torch
available. With the local venv, the GRPO Torch tests now run the real tensor and
optimizer paths instead of skipping them.

Torch also warned when NumPy was absent, so I added `numpy` to the optional
`training` extra. That makes the training environment less surprising for the
next person who installs it.

## What the loop does

The new module is `codeself.training.grpo_loop`. Its main function is
`run_grpo_training_loop`.

It accepts:

- an iterable of `TrainingBatch` objects;
- a callable that converts each batch into a differentiable `GRPOTensorBatch`;
- a Torch optimizer;
- an optional scheduler;
- loop config.

Then it:

- selects up to `max_batches`;
- groups microbatches for gradient accumulation;
- uses the effective accumulation size for final partial groups;
- calls the GRPO optimizer-step helper;
- steps the scheduler only when the optimizer steps;
- records per-microbatch and aggregate metrics.

This is still model-agnostic. The loop does not know how policy logprobs are
computed. That job belongs to the next layer, which will connect a causal LM or
toy policy module to the tensor batch builder.

## Why gradient accumulation matters now

Even for tiny code models, sequence length can make effective batch size
awkward. Gradient accumulation is the practical way to keep memory stable while
still training on grouped samples.

The loop chunks batches by `gradient_accumulation_steps`. If the last group is
smaller, it rescales by the actual group size. That keeps the final update from
being underweighted.

## What is still missing

The scaffold now has this chain:

```text
TrainingBatch -> GRPOTensorBatch -> GRPO loss -> optimizer step -> loop metrics
```

The next missing link is a model-facing tensor batch builder. It should run a
forward pass, gather token logprobs, attach old-policy and reference logprobs,
and hand the result to this loop. After that, CodeSelf will have the first real
end-to-end GRPO update path.
