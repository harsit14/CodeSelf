# Milestone 49: A Single-Config Online Training Launcher

Phase 8 now has a real launch surface.

The last milestone made online GRPO and PPO configs inspectable. That was useful,
but still incomplete: I could validate the shape of a config, but I could not
hand the repository one experiment file and ask it to assemble the actual run.

This milestone adds that missing layer.

## What landed

I added a new launcher module:

```text
src/codeself/training/online_launch.py
```

Its main entry points are:

```text
run_online_training_from_config()
run_online_training_from_config_file()
```

The launcher now assembles:

- train and optional eval tasks from JSONL registries;
- typed online GRPO/PPO configs;
- prompt templates;
- reward scorers;
- model, generator, and tokenizer surfaces;
- artifact directories;
- compact launch summaries.

The command-line entry point is:

```text
scripts/run_online_training.py
```

This is the first version of the command I actually want for experiments:

```text
python scripts/run_online_training.py --config configs/experiments/...
```

## The Model Backends

The launcher has two backend modes.

The first is `toy`. This is a tiny trainable Torch model with a bounded local
tokenizer. It exists so the full launch path can be tested quickly without model
downloads. It is not a research model and does not pretend to be one.

The second is `transformers`. For GRPO, the launcher can load a local
Transformers causal LM, adapt it to the rollout generator surface, and pass the
same policy model into the online trainer.

I added a GRPO launch template for an Apple Silicon local run:

```text
configs/experiments/grpo_online_transformers_launch.example.yaml
```

It uses:

```text
device: mps
training.device: mps
dtype: fp16
local_files_only: true
```

That matches the practical constraints of a MacBook Pro with 48 GB unified
memory: start with a small cached model, keep rollout sizes modest, and avoid
accidentally pulling weights during an experiment run.

## The Honest PPO Limitation

PPO is different. A plain causal LM does not provide the value estimates the PPO
path needs. The toy backend has a value surface, so PPO launcher smoke tests can
run end to end. But `model.backend=transformers` for PPO now fails early with a
clear value-head error instead of failing halfway through training.

That is the right failure mode. The next serious PPO step should add an explicit
value-head model integration rather than quietly pretending a language-model head
is a critic.

## Artifacts

Each launched run writes the existing cycle artifacts:

```text
cycle_0001/
  rollouts.jsonl
  metrics.jsonl
  checkpoint.json
  state/
```

It also writes:

```text
launch_summary.json
```

That file is intentionally compact. It records the algorithm, backend, rollout
mode, task counts, cycle counts, total rollouts, optimizer steps, and mean
reward. The heavy details still live in the cycle artifacts.

## What I Tested

The tests now cover:

- GRPO online launch from a single config mapping;
- PPO online launch through the new CLI;
- parsing the new launch example configs;
- rejecting PPO with plain Transformers until a value-head model path exists.

This closes the biggest remaining Phase 8 infrastructure gap. The scaffold can
now move from "I can express this run" to "I can launch this run," at least for
GRPO with local Transformers models and for fast local PPO smoke tests.
