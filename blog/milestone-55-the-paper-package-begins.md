# Milestone 55: The Paper Package Begins

Phase 9 starts with an unglamorous but important question:

Can someone open the repository and understand what CodeSelf actually is now?

After the GRPO, PPO, self-debug, online training, LoRA, and separate-critic
work, the old README was no longer honest. It still described the project like
a smoke-only scaffold. That was true at the beginning, but it undersold the
current shape of the system.

Today I rewrote the README and extended the reproducibility manifest.

## README Refresh

The README now describes the real architecture:

```text
Task JSONL
  -> prompt template
  -> generator or shared policy engine
  -> sandboxed execution
  -> reward scorer
  -> rollout JSONL and optional self-debug traces
  -> GRPO/PPO batch builder
  -> model-aware optimizer step
  -> cycle artifacts, evaluation reports, dashboards, manifests
```

It also points readers toward the right entry points:

- direct and self-debug rollout generation;
- GRPO/PPO smoke diagnostics;
- online training config inspection;
- toy and Transformers online launch templates;
- evaluation and paired-comparison scripts;
- dashboard rendering;
- reproducibility manifest generation.

The main goal was not marketing polish. It was orientation. A research repo
should make it easy to find the moving parts before asking anyone to trust the
numbers.

## Richer Reproducibility Manifests

The manifest already captured public source artifacts and git state. I extended
it with paper-package sections:

```text
config_artifacts
dataset_artifacts
environment_artifacts
model_references
checkpoint_manifests
checkpoint_artifacts
```

This gives us explicit SHA-256 hashes for experiment configs, dataset/split
artifacts, and environment lockfiles. It also parses model and tokenizer
references from known config paths such as:

```text
model.name
model.tokenizer
model.value_model.name
generation.model
```

Checkpoint manifests are handled separately. The archive still stays
conservative and public-source-oriented, but the manifest can summarize
checkpoint-manifest hashes and nested checkpoint artifact checksums when those
files exist.

That distinction matters. I do not want a convenience script silently bundling
private datasets or huge model weights. But I do want a paper run to have a
clear checksum trail.

## The Test Case

I added a miniature reproducibility test package:

- one experiment config with policy, tokenizer, value-model, and generation
  references;
- one split manifest;
- one environment file;
- one checkpoint manifest with a nested artifact checksum.

The test verifies JSON, Markdown, archive writing, model-reference extraction,
and checkpoint artifact extraction.

## Where This Leaves Phase 9

Phase 9 is now started rather than pending. The next paper-package work is less
about plumbing and more about presentation:

- result templates;
- plot placeholders;
- confidence-interval tables;
- failure-case sections;
- final claim checklists.

It is a different kind of engineering. Less tensor math, more making sure the
future reader does not have to reverse-engineer the experiment from a directory
tree at 2 a.m. Small mercy, honestly.
