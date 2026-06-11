# Milestone 27: Real State Checkpoints for the Toy GRPO Trainer

This milestone upgrades the minimal GRPO trainer from "metadata only" to actual
Torch state checkpointing.

The previous trainer could update model parameters, write metrics, and produce
a manifest. That was useful, but the manifest explicitly did not save model
state. This step adds the missing part: if the caller provides a state
directory, the trainer writes state files and records their checksums.

## What gets written

`run_grpo_model_training` now accepts `state_dir`.

When it is set, the trainer writes:

- `policy_model.pt`;
- `optimizer.pt`;
- `scheduler.pt`, if a scheduler with `state_dict` is supplied.

The checkpoint manifest includes a list of artifacts. Each artifact records:

- kind;
- path;
- byte size;
- SHA-256 digest.

That last field matters. Once checkpoints start mattering for experiments, the
manifest should make it obvious which exact state files produced a result.

## Keeping the boundary honest

This is still not a Transformers or LoRA checkpoint. The tests use the same tiny
causal-LM-like module as the previous milestone, but now the policy state is
saved with `torch.save` and loaded back during the test.

That keeps the checkpoint layer honest without downloading a model. The code now
knows how to persist state; the next layer can decide whether that state is a
full model, a PEFT adapter, or something else.

## The current chain

The Phase 5 chain now looks like:

```text
TrainingBatch -> model forward -> GRPO tensor batch -> optimizer step
  -> metrics -> state checkpoint -> manifest checksums
```

The next missing link is no longer the trainer skeleton. It is the real model
integration: Transformers loading, optional PEFT/LoRA attachment, and a tiny
local integration run that exercises the same path with an actual causal LM.
