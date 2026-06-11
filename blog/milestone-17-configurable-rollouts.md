# Milestone 17: Making Rollouts Configurable

The previous milestone added data quality checks and reward modes. This
milestone connects those pieces to the normal rollout path.

That connection matters because research code can become fragile when every
experiment is a slightly different command line. A reward mode should not be a
manual code edit. A data quality check should not be a separate ritual someone
remembers only on careful days. The experiment config should say what is being
run.

## A small config loader

CodeSelf still needs to preserve the dependency-free smoke path, so I did not
add Hydra or PyYAML here. Instead, I added a small config loader that supports
JSON and the simple YAML subset already used in the repository configs: nested
mappings, scalar values, and inline lists.

This is not meant to be a forever configuration system. It is meant to be good
enough for reproducible smoke and debug experiments while keeping CI light. If
the project later adopts Hydra, this loader can become the compatibility layer
for simple local runs.

## `run_rollouts.py --config`

The rollout script can now read an experiment config:

```bash
python3 scripts/run_rollouts.py \
  --config configs/experiments/rollout_smoke.example.yaml
```

The config can specify:

- task path, split, and limit;
- data quality behavior;
- prompt template;
- generation backend and sampling settings;
- execution settings;
- reward mode;
- output path.

CLI flags still work and override config values. That keeps the old workflow
intact while making scripted experiments easier to reproduce.

## Reward scorer injection

The rollout generator now accepts an injected reward scorer. By default it still
uses the original composite correctness reward, so existing tests and commands
keep their behavior. But a config or CLI flag can now select
`binary_all_tests_pass`, `fractional_pass_rate`, or `partial_credit`.

Each rollout also records reward metadata such as `reward_mode` and
`config_path`. This is a small detail, but it helps later when looking at JSONL
records and trying to remember exactly what experiment produced them.

## Data checks before generation

`run_rollouts.py` now runs the dataset quality checks before generating samples
unless the config or CLI explicitly disables them. If the task file leaks hidden
tests or duplicates train tasks into eval, the rollout job fails before spending
compute.

That is the right failure mode. In execution-feedback RL, bad data can look like
model improvement. I would rather make the pipeline a little stricter now than
debug a suspicious reward curve later.

## Why this milestone is a bridge

This is not the real training core yet. But it is the bridge to it. GRPO and PPO
will need the same choices: dataset split, prompt, generation settings, reward
mode, sandbox behavior, output paths, and seeds. Now there is a lightweight
pattern for reading those choices from a single experiment file.

The next implementation step is to give the smoke trainers and then the real
trainer skeleton the same config surface.
