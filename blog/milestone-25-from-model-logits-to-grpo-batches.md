# Milestone 25: From Model Logits to GRPO Batches

This milestone connects the GRPO loop to a model-facing tensor batch builder.

Up to this point, the training loop could optimize a prepared `GRPOTensorBatch`,
but something else had to manufacture that batch. That was useful for testing
the objective, but it was still one abstraction away from a real trainer. This
step adds the bridge from causal-LM logits to token-aligned GRPO tensors.

## Padding the training records

The new module is `codeself.training.grpo_model`. Its first job is boring but
necessary: turn a `TrainingBatch` into padded tensors.

It builds:

- `input_ids`;
- attention masks;
- response masks;
- per-sample advantages;
- optional old-policy logprobs;
- optional reference logprobs.

The response mask remains the central guardrail. Prompt and padding tokens are
allowed to exist in the tensor batch, but they do not contribute to the GRPO
loss.

## Gathering causal-LM logprobs

The helper `gather_causal_lm_token_logprobs` accepts logits shaped like:

```text
[batch, tokens, vocab]
```

It gathers next-token logprobs and returns a tensor aligned with the original
token positions. The first token receives logprob zero because there is no
previous context inside the packed sequence. Response token logprobs then line
up with the same response mask used by the loss.

This alignment is the detail I want nailed down before involving a real
Transformers model. Off-by-one errors in autoregressive logprobs are quiet, and
quiet errors are the worst kind in RL.

## A toy model, not a toy path

The tests use a tiny causal-LM-like parameter table instead of downloading a
model. It still has real `torch.nn.Parameter` values, real gradients, and a real
optimizer. The test runs:

```text
TrainingBatch -> model forward -> GRPOTensorBatch -> GRPO loop -> optimizer step
```

That is not paper training yet, but it is the first model-facing end-to-end
update path inside the scaffold.

## What comes next

The next layer should swap the toy model for a real causal-LM adapter. That
means attaching Transformers/PEFT, deciding where old-policy and reference
logprobs are cached, and adding checkpoint metadata. The core GRPO path is now
ready for that connection.
