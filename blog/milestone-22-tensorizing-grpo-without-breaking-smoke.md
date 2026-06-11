# Milestone 22: Tensorizing GRPO Without Breaking Smoke

This milestone continues Phase 5. The previous step gave CodeSelf a small,
dependency-free reference implementation of the GRPO objective. This step adds
the Torch version that can actually participate in backpropagation.

The constraint was simple: the default smoke path still cannot require Torch.
The repo should remain pleasant to test on a fresh laptop checkout, while the
real training path can opt into the heavier stack.

## A lazy Torch boundary

I added `codeself.training.grpo_torch`. It exports:

- `GRPOTensorBatch`;
- `GRPOTensorLossResult`;
- `build_grpo_tensor_batch`;
- `compute_grpo_tensor_loss`;
- `torch_training_available`;
- `require_torch`.

The module does not import Torch at module import time. That means
`import codeself.training` remains safe even when the training extra has not
been installed. Torch is imported only when a caller asks to build tensor
batches or compute the tensor loss.

This keeps the MacBook M5 Pro development loop fast. The normal unit suite can
still run without pulling in model-training dependencies, and the Torch-specific
tests become active automatically in environments that have Torch installed.

## Padded tensor batches

The new batch builder converts a `TrainingBatch` into padded tensors:

- current policy logprobs;
- old-policy logprobs;
- optional reference logprobs;
- response masks;
- per-sample advantages.

Padding matters because generated code responses have different lengths. The
response mask is the guardrail: padded positions and prompt positions should not
contribute to the loss.

## Matching the reference reduction

One small design choice mattered more than it first appeared. The float
reference implementation computes a mean over response tokens per sample, then
averages samples. The first tensor sketch could have averaged all response
tokens in the batch at once, which would weight longer generations more heavily.

I aligned the Torch path with the reference: sequence mean first, batch mean
second. That makes diagnostics comparable between the plain Python reference
and the differentiable Torch loss.

## What is now differentiable

`compute_grpo_tensor_loss` computes:

- the clipped GRPO surrogate loss;
- optional reference-model KL penalty;
- a final scalar `loss` tensor;
- detached diagnostics for ratio, clipped ratio, approximate KL, and clipped
  token fraction.

This is still not a trainer. There is no optimizer step yet, no LoRA adapter
attachment, no gradient accumulation, and no checkpoint save. But the core
objective now exists in the form a real training loop needs.

The next milestone should connect this tensor objective to a minimal trainer
step: build a tensor batch, compute loss, call backward, step an optimizer, and
emit metrics that line up with the reference diagnostics.
