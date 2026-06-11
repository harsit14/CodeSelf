# Milestone 26: A Minimal Model-Aware GRPO Trainer

This milestone adds a small trainer around the GRPO loop.

The previous milestone connected model logits to GRPO tensor batches. That made
the objective model-facing, but the caller still had to create an optimizer,
configure the loop, wire the batch builder, and decide where metrics should go.
This milestone packages those steps into a minimal trainer.

## What the trainer owns

The new module is `codeself.training.grpo_trainer`. It exports:

- `GRPOModelTrainingConfig`;
- `GRPOModelTrainingResult`;
- `run_grpo_model_training`.

The runner accepts `TrainingBatch` records, a policy model, optional old-policy
and reference models, and optional optimizer/scheduler objects. If no optimizer
is supplied, it creates AdamW from the trainable policy parameters.

It also sets model modes:

- policy model to train mode;
- old-policy and reference models to eval mode.

That is a small detail, but it keeps the future Transformers integration from
having to rediscover the same convention.

## Metrics and checkpoint metadata

The trainer can now write two lightweight artifacts:

- a JSONL metrics file, one row per loop step;
- a checkpoint manifest with config, model class names, optimizer class, and
  loop summary.

This manifest is not a model checkpoint yet. It does not save LoRA weights or
full model weights. It is deliberately honest metadata around a tiny run. The
next real checkpointing step should add adapter/model state separately and then
record checksums.

## Why this is enough for now

The current end-to-end path is:

```text
TrainingBatch -> policy forward -> GRPOTensorBatch -> GRPO loop -> optimizer
```

The tests use a tiny causal-LM-like module with a real `torch.nn.Parameter`.
The trainer updates that parameter, writes metrics, and writes a manifest. That
is not a paper experiment, but it is the first complete model-update circuit in
the scaffold.

The next milestone should replace the toy module with a real local causal-LM
adapter: Transformers loading, PEFT/LoRA attachment, and model weight or adapter
checkpoint writing.
