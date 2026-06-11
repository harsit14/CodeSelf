# Milestone 23: A Minimal GRPO Optimizer Step

This milestone adds the smallest useful training-step primitive on top of the
tensor GRPO objective.

The previous milestone made the loss differentiable. That is necessary, but not
quite training. A real trainer still needs to call backward, handle gradient
accumulation, clip gradients, step an optimizer, and record metrics. This
milestone implements that narrow bridge.

## What the step owns

The new module is `codeself.training.grpo_step`. It exports:

- `GRPOOptimizerStepConfig`;
- `GRPOOptimizerStepResult`;
- `run_grpo_optimizer_step`.

The function accepts a prepared `GRPOTensorBatch` and a Torch optimizer. It then:

- computes the tensor GRPO loss;
- divides the loss by `gradient_accumulation_steps`;
- calls `backward`;
- optionally clips gradients;
- optionally calls `optimizer.step`;
- returns detached diagnostics.

The diagnostics include loss, backward loss, policy loss, KL loss, gradient
norm, response tokens, mean ratio, approximate KL, and clipped-token fraction.

## Still lazy, still optional

Torch is still imported lazily. If the training extra is not installed, the
default unit suite still runs. The tests check the missing-dependency path now,
and the real optimizer-update tests will automatically run in an environment
where Torch is available.

That matters because I want two loops to coexist:

- the lightweight loop on the MacBook M5 Pro for contracts and smoke tests;
- the heavier local or GPU loop for actual model training.

The codebase should not punish the first loop just because the second exists.

## What this does not own

This step does not run a model forward pass. It does not generate rollouts. It
does not compute fresh policy logprobs from hidden states. It also does not
attach LoRA adapters, step a scheduler, or write checkpoints.

That separation is intentional. The optimizer step should be boring and
inspectable. The next layer can be responsible for assembling the ingredients:
model engine, tokenizer, rollout records, execution rewards, old/reference
logprobs, and finally this step.

## Why this milestone matters

The scaffold now has a chain:

```text
TrainingBatch -> padded tensors -> GRPO tensor loss -> optimizer step
```

The missing piece is no longer the core objective or the backward step. The next
useful milestone is to build a tiny trainer loop around this chain, even if it
starts with a toy model or cached tiny LM.
