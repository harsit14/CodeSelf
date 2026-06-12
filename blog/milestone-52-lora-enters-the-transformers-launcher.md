# Milestone 52: LoRA Enters the Transformers Launcher

This milestone removes an awkward contradiction from the Phase 8 launcher.

The plan said local small-model work should default to LoRA, especially on a
MacBook with 48GB unified memory. The code, however, still rejected
`model.use_lora=true` for the single-config online launcher. That meant the
most realistic local path was documented, but not actually wired.

Today I turned that guardrail into a real adapter assembly path.

## What Landed

`ModelRuntimeConfig` now carries the practical LoRA knobs:

```text
lora_rank
lora_alpha
lora_dropout
lora_target_modules
```

The config loader preserves them in the normalized training-core payload, so
debug scripts can show the exact adapter shape before a run starts.

The main behavior change is in `TransformersModelEngine`: when
`model.use_lora=true`, it lazily imports PEFT, builds a `LoraConfig`, and wraps
the loaded causal LM with `get_peft_model()`.

That lazy boundary matters. Dependency-free tests and smoke paths still do not
import Torch, Transformers, or PEFT. Real model runs only pay for PEFT when the
experiment config asks for adapters.

## Why the Engine Owns It

I put the LoRA attachment in the model engine instead of only in the online
launcher. The reason is simple: the same policy object has to serve two roles.

It generates rollouts through `ModelEngineCodeGenerator`.

It receives GRPO/PPO optimizer updates during training.

If the launcher wrapped only the training model, generation would still sample
from the base model. That would make online RL subtly wrong: the policy being
optimized would not be the policy collecting the next batch of executions.

By adapting the engine's underlying model, both paths share the same LoRA
weights.

## PPO Value Heads

PPO still uses the integrated value-head wrapper by default. The order now is:

```text
base causal LM -> PEFT LoRA policy -> PPO value-head wrapper
```

Old-policy and old-value snapshots can be assembled with the same LoRA
architecture, so state syncing keeps working across online cycles.

Reference models default to the frozen base causal LM. That is the right
default for KL: the reference should describe the pre-RL policy, not another
trainable adapter copy.

## Config Templates

I updated the GRPO and PPO Transformers launch examples to use LoRA by default:

```text
use_lora: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
full_finetune: false
```

That is a much better default for local experiments on this machine. Full
fine-tuning remains possible, but it should be an explicit choice rather than
the only launchable path.

## What I Tested

The new tests cover:

- config parsing for alpha, dropout, and target modules;
- early validation of invalid adapter settings;
- lazy PEFT config construction with a fake `peft` module;
- the offline PPO Transformers launch path with `use_lora=true`.

This closes the LoRA part of the remaining Phase 8 launcher risk. The next
open launcher question is the separately pretrained PPO critic/value model,
which is more of an architecture choice than a config plumbing issue.
