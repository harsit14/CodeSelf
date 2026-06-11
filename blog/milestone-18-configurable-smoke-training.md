# Milestone 18: Giving Smoke Training The Same Config Surface

This milestone makes the GRPO and PPO smoke scripts behave more like real
experiments.

They still do not train a model. That line remains explicit. But they now read
the same kinds of configuration as rollout generation: data path, split,
generation settings, execution settings, reward mode, output directory, and
quality-check behavior.

## Why this matters before real training

The next major jump is a real training core. Before writing that code, I want
the existing smoke path to answer a practical question:

Can an experiment be described by one config file and then run through the
pipeline without hidden command-line state?

For rollouts, the answer became yes in the last milestone. Now GRPO and PPO
smoke runs are catching up.

## Reward scorers are now injectable

The smoke trainers used to call `generate_rollouts()` with the default
correctness reward baked in. They now accept an optional reward scorer and pass
it through to rollout generation.

That means a GRPO smoke run can use:

- the original `correctness_v0` reward;
- a sparse binary all-tests-pass reward;
- a fractional pass-rate reward;
- a partial-credit reward with optional shaping.

This is still diagnostic, not optimization. But it lets the smoke metrics
exercise the same reward choices that the real trainer will eventually optimize.

## Config-driven scripts

Both smoke scripts now accept `--config`:

```bash
python3 scripts/train_grpo_smoke.py \
  --config configs/experiments/grpo_smoke.local.example.json

python3 scripts/train_ppo_smoke.py \
  --config configs/experiments/ppo_smoke.local.example.json
```

CLI flags still override config values, which is useful for quick local
experiments. For example, I can run the bundled config but redirect outputs to
`/tmp` and cap the run at one step.

The checkpoint manifests now include the active `reward_mode` and `config_path`.
That is a small but important reproducibility detail. If a rollout JSONL file or
checkpoint manifest survives longer than my memory, it should still tell me how
it was produced.

## Data quality checks before smoke training

The GRPO/PPO smoke scripts now run the same dataset quality checks as
`run_rollouts.py`. If there is hidden-test leakage or train/eval duplication,
the script fails before generating training diagnostics.

This is slightly stricter than the original smoke scripts, but it is the kind of
strictness I want before adding model updates. Smoke tests should be permissive
about dependencies, not permissive about contaminated experiment inputs.

## What comes next

At this point, the dependency-free path has a coherent config and reward
surface:

- rollout generation can read configs;
- GRPO smoke can read configs;
- PPO smoke can read configs;
- all three can use pluggable reward modes;
- all three can run data-quality checks.

The next phase should start the real training-core skeleton: model/tokenizer
interfaces, prompt/response masks, rollout tensors, logprob records, reference
model contracts, and trainer backend selection. That will finally create the
space where actual GRPO and PPO implementations can land.
