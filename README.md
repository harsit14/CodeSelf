# CodeSelf

CodeSelf is a research scaffold for execution-feedback reinforcement learning
on Python coding tasks. The central question is deliberately bounded: can a
coding policy improve on a declared task distribution after practicing against
sandboxed tests and structured feedback?

The repository now contains an end-to-end local research loop:

- canonical task ingestion with public and hidden tests;
- sandboxed execution and correctness-first reward modes;
- direct and self-debug rollout generation;
- GRPO and PPO objectives with Torch-backed model-training paths;
- online collect -> execute -> score -> optimize cycles;
- Transformers launch templates with LoRA, PPO value heads, and separate PPO
  critic support;
- evaluation reports, paired statistics, and static HTML dashboards;
- reproducibility manifests with config, dataset, environment, model, and
  checkpoint metadata.

This is still a scaffold, not a released trained model. The dependency-free
smoke path is meant to stay fast and reviewable, while optional training
dependencies unlock tiny/local model integration and single-GPU experiments.

## Architecture

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

Core packages:

```text
src/codeself/datasets/      Task schema, loaders, split manifests, quality checks.
src/codeself/execution/     Static scan, subprocess runner, Docker runner scaffold.
src/codeself/rewards/       Correctness and configurable reward modes.
src/codeself/agent/         Prompting, parsing, rollouts, self-debug traces.
src/codeself/training/      Common records, GRPO/PPO losses, online training loops.
src/codeself/evaluation/    pass@k, paired stats, learning curves, dashboard renderer.
src/codeself/reporting/     Reproducibility manifest and archive helpers.
```

## Repository Layout

```text
configs/       Dataset, reward, model, rollout, online training, and eval examples.
data/          Local dataset placeholders; raw and processed data are git-ignored.
docker/        Container execution scaffold for locked-down evaluation.
docs/          Reproducibility, model-card, dataset-card, plan, and safety docs.
outputs/       Local rollout/checkpoint/report placeholders; generated files are ignored.
scripts/       Command-line entry points for the pipeline.
src/           CodeSelf Python package.
tests/         Unit tests for smoke, training, evaluation, launcher, and reporting paths.
blog/          Milestone implementation log, force-added when a milestone ships.
```

## Quick Start

Use Python 3.11 or newer. The smoke path has no required third-party
dependencies.

```bash
python3 -m unittest discover -s tests
```

Validate the bundled example task:

```bash
python3 scripts/validate_task_schema.py configs/datasets/tasks.example.jsonl
```

Generate and evaluate direct mock rollouts:

```bash
python3 scripts/run_rollouts.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/mock_smoke.jsonl \
  --backend mock \
  --samples-per-task 4

python3 scripts/evaluate_rollouts.py \
  --rollouts outputs/rollouts/mock_smoke.jsonl \
  --output outputs/reports/mock_baseline.md \
  --power-output outputs/reports/mock_power.json
```

Run self-debug rollouts with trace output:

```bash
python3 scripts/run_rollouts.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/self_debug_smoke.jsonl \
  --rollout-mode self_debug \
  --trace-output outputs/reports/self_debug_traces.jsonl \
  --backend mock \
  --samples-per-task 2 \
  --max-revisions 1
```

## Training Paths

Fast smoke diagnostics remain available:

```bash
python3 scripts/train_grpo_smoke.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output-dir outputs/checkpoints/grpo_smoke_example \
  --backend mock \
  --group-size 4 \
  --max-steps 3

python3 scripts/train_ppo_smoke.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output-dir outputs/checkpoints/ppo_smoke_example \
  --backend mock \
  --samples-per-task 4 \
  --max-steps 3
```

Typed online training configs can be inspected without loading model weights:

```bash
python3 scripts/inspect_online_training_config.py \
  --config configs/experiments/grpo_online_self_debug.example.yaml

python3 scripts/inspect_online_training_config.py \
  --config configs/experiments/ppo_online_transformers_separate_value.example.yaml
```

The single-config launcher is:

```bash
python3 scripts/run_online_training.py \
  --config configs/experiments/ppo_online_toy_launch.example.yaml
```

For real local model experiments, start from:

```text
configs/experiments/grpo_online_transformers_launch.example.yaml
configs/experiments/ppo_online_transformers_launch.example.yaml
configs/experiments/ppo_online_transformers_separate_value.example.yaml
```

Those templates assume cached Hugging Face weights, Apple Silicon-friendly
`mps`/`fp16` defaults, and LoRA enabled. Pin exact model and tokenizer revisions
before treating any run as reproducible.

## Evaluation And Dashboarding

Evaluate rollout files:

```bash
python3 scripts/evaluate_rollouts.py \
  --rollouts outputs/rollouts/mock_smoke.jsonl \
  --output outputs/reports/mock_eval.md \
  --json-output outputs/reports/mock_eval.json \
  --group-by metadata.rollout_mode
```

Compare two rollout files with paired statistics:

```bash
python3 scripts/compare_rollouts.py \
  --baseline outputs/rollouts/baseline.jsonl \
  --candidate outputs/rollouts/candidate.jsonl \
  --output outputs/reports/comparison.md \
  --json-output outputs/reports/comparison.json
```

Render a portable dashboard:

```bash
python3 scripts/render_evaluation_dashboard.py \
  --evaluation baseline=outputs/reports/mock_eval.json \
  --output outputs/reports/dashboard.html
```

Online training artifact directories can also be passed as learning curves with
`--curve label=artifacts/run_name`.

## Reproducibility

Generate the public reproducibility manifest:

```bash
python3 scripts/make_reproducibility_manifest.py \
  --output outputs/reports/reproducibility_manifest.json \
  --markdown-output outputs/reports/reproducibility_manifest.md \
  --archive outputs/reports/codeself_reproducibility.tar.gz
```

The manifest records:

- git revision and dirty status;
- public source/config artifact hashes;
- experiment config hashes;
- dataset and split artifact hashes;
- environment lockfiles such as `pyproject.toml`;
- model/tokenizer references parsed from configs;
- checkpoint manifest hashes and checkpoint artifact checksums when present;
- rerun commands for the smoke pipeline.

Release-facing documents:

- [Reproducibility checklist](./docs/reproducibility_checklist.md)
- [Adapter model-card template](./docs/model_card_adapters.md)
- [Private task dataset-card template](./docs/dataset_card_private_tasks.md)
- [Limitations and safety notes](./docs/limitations_and_safety.md)
- [Overhaul plan](./docs/overhaul_plan.md)

## Optional Dependencies

The base package intentionally has no third-party runtime dependencies.
Install optional training/reporting dependencies when needed:

```bash
pip install -e '.[training,reporting]'
```

The `training` extra lists Torch, Transformers, PEFT, Accelerate, TRL, vLLM,
and related libraries. The `reporting` extra lists heavier analysis tools.

## Safety

Generated code is untrusted. Keep hidden tests out of prompts and public traces,
execute generated solutions only in a restricted environment, and reserve locked
private test sets for final evaluation rather than checkpoint selection.

## License

MIT. See [LICENSE](./LICENSE).
