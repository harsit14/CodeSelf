# Milestone 34: Tensorizing PPO

The previous milestone made PPO's scalar objective testable. This one makes it
differentiable.

The new module is `ppo_torch.py`. It mirrors the GRPO tensor path: lazy Torch
imports, padded tensor batches, response-token masking, and detached diagnostics
that can be compared against the dependency-free reference implementation.

## What landed

`build_ppo_tensor_batch` converts `TrainingBatch` records plus PPO value
estimates into padded tensors for:

- policy logprobs;
- old-policy logprobs;
- response masks;
- normalized advantages;
- returns;
- current values;
- old values;
- entropy;
- optional reference logprobs.

`compute_ppo_tensor_loss` then computes the clipped PPO objective with:

- clipped policy loss;
- clipped value loss;
- entropy bonus;
- optional reference KL;
- response-only sequence means.

The tests compare tensor losses against the scalar reference, check
variable-length padding, and verify gradients flow through policy logprobs and
value predictions.

## Why this slice matters

PPO needs a value head, so the tensor path has more moving pieces than GRPO.
By landing the padded objective first, the next milestone can focus on model
wiring instead of debugging loss math and padding at the same time.
