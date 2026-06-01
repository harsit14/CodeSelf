# CodeSelf

CodeSelf is a research scaffold for studying execution-feedback reinforcement
learning on coding tasks. It focuses on a bounded question: can a coding model
improve on a declared Python task distribution after practicing with sandboxed
test feedback?

The repository includes the infrastructure needed to run the smoke pipeline end
to end: dataset conversion, sandboxed execution, correctness-first rewards,
rollout records, baseline evaluation, GRPO/PPO diagnostics, paired statistical
testing, agentic self-debug traces, and reproducibility manifests.

This is not a released trained model. The GRPO and PPO trainers in this public
version are dependency-free smoke scaffolds that validate the research workflow
without updating model weights.

## What Is Included

- Canonical task schema with public/hidden tests and deterministic split helpers.
- MBPP-style, HumanEval/EvalPlus-style, and canonical JSONL dataset loaders.
- Conservative Python security scan plus subprocess execution runner.
- Task runner with syntax, import, public-test, and hidden-test phases.
- `reward_v0_correctness`, a correctness-first reward with quality and efficiency diagnostics.
- Prompt templates, code parsing, mock/static/local generation backends, and rollout JSONL files.
- Baseline evaluation with pass@k, parser failure rate, reward stats, and power-planning diagnostics.
- GRPO smoke diagnostics with grouped reward advantages and informative-prompt tracking.
- PPO smoke diagnostics with value-baseline advantages and clipping diagnostics.
- Final paired comparison tools with exact McNemar/binomial tests and bootstrap intervals.
- Agentic self-debug loop traces for public-test use, revision behavior, and final submission.
- Reproducibility checklist, model-card template, dataset-card template, and manifest/archive tool.

## Repository Layout

```text
configs/       Example dataset, model, reward, and experiment configs.
data/          Local dataset placeholders; raw/processed data is git-ignored.
docker/        Final-evaluation executor scaffold.
docs/          Reproducibility, model-card, dataset-card, and safety docs.
outputs/       Local rollout/checkpoint/report placeholders; generated files are git-ignored.
scripts/       Command-line entry points for the pipeline.
src/codeself/  Python package for datasets, execution, rewards, agents, training, and reporting.
tests/         Unit tests for the public smoke implementation.
```

## Quick Start

Use Python 3.11 or newer. The smoke pipeline has no required third-party
dependencies.

```bash
python3 -m unittest discover -s tests
```

Validate the bundled example task:

```bash
python3 scripts/validate_task_schema.py configs/datasets/tasks.example.jsonl
```

Generate and evaluate mock rollouts:

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

Run GRPO and PPO smoke diagnostics:

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

python3 scripts/compare_training_smoke.py \
  --grpo-metrics outputs/checkpoints/grpo_smoke_example/metrics.jsonl \
  --ppo-metrics outputs/checkpoints/ppo_smoke_example/metrics.jsonl \
  --output outputs/reports/ppo_vs_grpo_report.md
```

## Dataset Conversion

Convert a local MBPP-style file into the canonical CodeSelf task format:

```bash
python3 scripts/prepare_datasets.py \
  --format mbpp \
  --input path/to/mbpp.json \
  --output data/processed/mbpp.codeself.jsonl \
  --assign-splits \
  --manifest data/splits/mbpp.manifest.json
```

Hidden tests should be stored outside prompts and starter code. The validation
script checks for common leakage mistakes in canonical task files.

## Agentic Traces

Run the self-debug smoke loop with public-test feedback:

```bash
python3 scripts/run_agentic.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --trace-output outputs/reports/agentic_traces.jsonl \
  --report-output outputs/reports/agentic_strategy_report.md \
  --backend mock \
  --max-revisions 1
```

Render one trace:

```bash
python3 scripts/view_agent_trace.py \
  --traces outputs/reports/agentic_traces.jsonl \
  --output outputs/reports/agent_trace.md
```

## Reproducibility

The public methodology is in [methodology.md](./methodology.md). Additional
release documents are in [docs/](./docs):

- [Reproducibility checklist](./docs/reproducibility_checklist.md)
- [Adapter model-card template](./docs/model_card_adapters.md)
- [Private task dataset-card template](./docs/dataset_card_private_tasks.md)
- [Limitations and safety notes](./docs/limitations_and_safety.md)

Create a reproducibility manifest and archive for the public project surface:

```bash
python3 scripts/make_reproducibility_manifest.py \
  --output outputs/reports/reproducibility_manifest.json \
  --markdown-output outputs/reports/reproducibility_manifest.md \
  --archive outputs/reports/codeself_reproducibility.tar.gz
```

## Optional Training Dependencies

The `training` extra in `pyproject.toml` lists libraries expected for future
real model training, including Transformers, TRL, PEFT, Accelerate, and vLLM.
Those dependencies are not required for the smoke pipeline.

## Safety

Generated code is untrusted. Run generated solutions in an isolated environment,
keep hidden tests out of prompts and public traces, and reserve private test
sets for locked final evaluation only.

## License

MIT. See [LICENSE](./LICENSE).
