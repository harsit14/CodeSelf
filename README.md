# CodeSelf

CodeSelf is a Python research project for training coding agents with
execution feedback. It generates solutions for programming tasks, runs them in
a restricted sandbox, scores the results, and turns those rollouts into GRPO or
PPO training batches.

The goal is practical and measurable: can a small coding model improve after
practicing on a known task distribution with real tests and reward signals?

## What It Does

| Stage | What happens |
| --- | --- |
| Dataset | Load Python coding tasks from canonical JSONL, MBPP-style files, or HumanEval-style files. |
| Generation | Produce candidate Python solutions with a mock, static, or Transformers-backed generator. |
| Execution | Run syntax checks, static security checks, public tests, and hidden tests in an isolated subprocess. |
| Rewards | Score correctness, partial credit, failures, parse errors, and execution outcomes. |
| Training | Build GRPO/PPO batches with response-token masks, advantages, KL penalties, and optimizer steps. |
| Evaluation | Report pass@k, reward summaries, paired comparisons, learning curves, and static dashboards. |

## Results Snapshot

These tables are small sanity checks, not public benchmark claims. They are
included so readers can quickly understand what the pipeline produces.

### No-Dependency Smoke Run

Reproducible with the checked-in example task and mock generator:

```bash
python3 scripts/run_rollouts.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/readme_smoke_mock.jsonl \
  --backend mock \
  --samples-per-task 4 \
  --seed 20260601 \
  --skip-data-quality-checks

python3 scripts/evaluate_rollouts.py \
  --rollouts outputs/rollouts/readme_smoke_mock.jsonl \
  --output outputs/reports/readme_smoke_eval.md \
  --ks 1,2,4
```

| Metric | Value |
| --- | ---: |
| Tasks | 1 |
| Rollouts | 4 |
| Execution pass rate | 75% |
| Mean reward | 0.85 |
| Parse failure rate | 0% |
| Degenerate output rate | 0% |
| pass@1 | 75% |
| pass@2 | 100% |
| pass@4 | 100% |

### Small Debug Training Run

The debug configs train on the self-contained `codeself-debug` dataset with 8
training tasks and 2 dev tasks. The run below used the provided
Transformers + LoRA path for 15 online cycles.

| Algorithm | Train reward | Train pass rate | Dev reward | Dev pass@1 | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| GRPO | 0.33 -> 0.74 | 38% -> 66% | 0.72 -> 0.84 | 75% -> 88% | Clear improvement on the debug run. |
| PPO | 0.38 -> 0.26 | 44% -> 30% | 0.49 -> 0.46 | 50% -> 50% | Stable, but no lift in this tiny setting. |

Use these numbers as a pipeline demonstration. For benchmark-grade claims, run
larger datasets, fixed model revisions, repeated seeds, and held-out evaluation.

## Architecture

```text
Task JSONL
  -> prompt template
  -> code generator or model engine
  -> sandboxed execution
  -> reward scorer
  -> rollout JSONL
  -> GRPO / PPO batch builder
  -> optimizer step
  -> evaluation reports and dashboard
```

Core package layout:

| Path | Purpose |
| --- | --- |
| `src/codeself/datasets/` | Task schemas, loaders, split manifests, quality checks. |
| `src/codeself/execution/` | Security scan, subprocess sandbox, Docker runner, per-test outcomes. |
| `src/codeself/rewards/` | Correctness and configurable reward modes. |
| `src/codeself/agent/` | Prompts, parsing, rollout records, self-debug traces. |
| `src/codeself/training/` | GRPO/PPO losses, model wrappers, online cycles, smoke trainers. |
| `src/codeself/evaluation/` | pass@k, paired statistics, learning curves, dashboard rendering. |
| `src/codeself/tracking/` | JSONL, TensorBoard, and W&B metric adapters. |
| `src/codeself/reporting/` | Reproducibility manifests and artifact summaries. |

## Quickstart

Python 3.11+ is recommended. The smoke path uses only the standard library.

```bash
git clone https://github.com/harsit14/CodeSelf.git
cd CodeSelf

python3 scripts/run_rollouts.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/mock.jsonl \
  --backend mock \
  --samples-per-task 4 \
  --skip-data-quality-checks

python3 scripts/evaluate_rollouts.py \
  --rollouts outputs/rollouts/mock.jsonl \
  --output outputs/reports/mock_eval.md \
  --ks 1,2,4
```

For development checks:

```bash
python3 -m unittest \
  tests.unit.test_execution \
  tests.unit.test_rewards \
  tests.unit.test_evaluation \
  tests.unit.test_training_common
```

## Optional Training Setup

Install the optional training dependencies when you want to use real models:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[training]'
```

The debug configs default to `Qwen/Qwen3-0.6B-Base` and can also point at any
compatible causal language model path.

Build the debug dataset and start a GRPO run:

```bash
python3 scripts/build_dataset.py \
  --config configs/datasets/debug.dataset.yaml \
  --output data/processed/debug_tasks.jsonl \
  --manifest data/splits/debug_manifest.json

python3 scripts/run_online_training.py \
  --config configs/experiments/grpo_debug_local.yaml
```

Run the matching PPO comparison:

```bash
python3 scripts/run_online_training.py \
  --config configs/experiments/ppo_debug_local.yaml
```

## Reports And Dashboards

Create an evaluation report:

```bash
python3 scripts/evaluate_rollouts.py \
  --rollouts outputs/rollouts/mock.jsonl \
  --output outputs/reports/mock_eval.md \
  --ks 1,2,4
```

Render a static dashboard from rollout or evaluation files:

```bash
python3 scripts/render_evaluation_dashboard.py \
  --rollouts smoke=outputs/rollouts/mock.jsonl \
  --output outputs/reports/dashboard.html
```

Generate a reproducibility manifest:

```bash
python3 scripts/make_reproducibility_manifest.py \
  --output outputs/reports/reproducibility_manifest.json \
  --markdown-output outputs/reports/reproducibility_manifest.md
```

## Dataset Format

A minimal task looks like this:

```json
{
  "task_id": "toy/add-one",
  "source": "example",
  "prompt": "Write a function add_one(x) that returns x + 1.",
  "split": "train",
  "entry_point": "add_one",
  "public_tests": [
    {"name": "public-basic", "code": "assert add_one(1) == 2"}
  ],
  "hidden_tests": [
    {"name": "hidden-zero", "code": "assert add_one(0) == 1"}
  ]
}
```

The loader also supports MBPP-style and HumanEval-style inputs through
`scripts/prepare_datasets.py`.

## Safety

Generated code is untrusted. CodeSelf runs candidates in a restricted
subprocess with static checks, runtime hardening, CPU/file/process limits,
network blocking, per-phase isolation, and a Docker runner for stricter final
evaluation environments. Hidden tests should stay out of prompts, public traces,
and model inputs.

## License

MIT. See [LICENSE](./LICENSE).
