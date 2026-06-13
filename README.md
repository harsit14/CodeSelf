# CodeSelf

CodeSelf is a runnable research framework for **execution-feedback reinforcement
learning** on Python coding tasks. The central question is deliberately bounded:

> Can a small coding policy measurably improve on a declared task distribution
> after practicing against sandboxed tests and structured feedback?

The framework runs end to end on a single GPU (or an Apple-Silicon laptop):
dataset ingestion with a contamination gate, a hardened sandbox, correctness-first
rewards with reward-hacking guards, batched rollout sampling, **real GRPO and PPO
LoRA training** on an open model, held-out evaluation with paired statistics,
experiment tracking, dashboards, and reproducibility manifests.

A dependency-free **smoke path** stays fast for CI; installing the optional
training extra unlocks real model training.

## Research question and result

On the bundled 12-task `codeself-debug` distribution, training
**Qwen2.5-Coder / Qwen3-0.6B-class** policies with LoRA for 15 online cycles on
an Apple M-series laptop (MPS):

| Metric | GRPO | PPO (same protocol) |
| --- | --- | --- |
| Train reward (first 3 → last 3 cycles) | 0.42 → **0.74** | 0.35 → 0.24 |
| Sampled pass-rate (temp 1.0) | 38% → **69%** | ~35% (flat) |
| Greedy pass@1 (held-out) | 83% → **92%** | — |
| KL to frozen reference | 0 → 0.087 | finite, stable |

GRPO moves the policy clearly; PPO is stable (finite grad norms, decreasing
value loss, no collapse) but does not improve under the identical small-batch,
cold-critic protocol — a reproducible algorithm comparison, consistent with
PPO's higher sample/warmup requirements for LLM RL.

> These are debug-scale numbers for validating the pipeline, not a benchmark
> claim. Scale the dataset and cycles for paper-grade results.

## Architecture

```text
Task JSONL  (versioned dataset config + contamination gate)
   │
   ├─ prompt template ─► batched rollout sampling (policy engine)
   │                         │
   │                    sandboxed execution (process jail, rlimits, no net)
   │                         │
   │                    reward scorer (+ reward-hacking flags)
   │                         │
   │                    rollout records / self-debug traces
   │                         │
   │                ┌────────┴─────────┐
   │            GRPO batch          PPO batch
   │          group-relative      GAE + value head
   │          advantages          clipped value loss
   │                └────────┬─────────┘
   │                  microbatched forward+backward (memory-bounded)
   │                  clipped PG loss + KL to frozen reference
   │                         │
   └──────────────►  online cycle: collect → execute → score → optimize
                             │
                       checkpoints · metrics · dev eval
                             │
        pass@k · paired bootstrap / permutation · effect sizes
                             │
              dashboards · trackers (W&B/TensorBoard/JSONL) · manifests
```

Core packages:

```text
src/codeself/datasets/     Task schema, loaders, versioned dataset configs, contamination gate.
src/codeself/execution/    Static scan, subprocess jail, Docker runner, per-test outcomes.
src/codeself/rewards/      Correctness/configurable rewards, reward-hacking detection.
src/codeself/agent/        Prompting, parsing, batched rollouts, self-debug traces.
src/codeself/training/     GRPO/PPO losses, model-forward batches, microbatched loops, online cycles.
src/codeself/evaluation/   pass@k, paired stats, learning curves, dashboard renderer.
src/codeself/tracking/     Pluggable experiment trackers (W&B / TensorBoard / JSONL).
src/codeself/reporting/    Reproducibility manifest with hardware + library capture.
```

## Install

Python 3.11+. The smoke path needs no third-party dependencies.

```bash
# Smoke / CI only:
python3 -m unittest discover -s tests

# Real training (Torch + Transformers + PEFT + Accelerate):
pip install -e '.[training]'

# Optional: experiment tracking and heavier analysis
pip install -e '.[tracking]'    # wandb, tensorboard
pip install -e '.[reporting]'   # pandas, plotly, scipy, duckdb
```

`vllm` is a separate extra (`.[vllm]`) because it does not build on macOS/MPS.

## Quickstart

### 1. Smoke (no model, no GPU — the CI path)

```bash
python3 -m unittest discover -s tests
python3 scripts/run_rollouts.py --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/mock.jsonl --backend mock --samples-per-task 4
python3 scripts/evaluate_rollouts.py --rollouts outputs/rollouts/mock.jsonl \
  --output outputs/reports/mock_eval.md
```

### 2. Debug (real model, ~10 tasks, single GPU / MPS)

Build the self-contained debug dataset, then train:

```bash
python3 scripts/build_dataset.py \
  --config configs/datasets/debug.dataset.yaml \
  --output data/processed/debug_tasks.jsonl \
  --manifest data/splits/debug_manifest.json

python3 scripts/run_online_training.py \
  --config configs/experiments/grpo_debug_local.yaml      # GRPO
python3 scripts/run_online_training.py \
  --config configs/experiments/ppo_debug_local.yaml       # PPO comparison
```

Self-debug ablation (generate → execute → revise):

```bash
python3 scripts/run_online_training.py \
  --config configs/experiments/grpo_debug_local_self_debug.yaml
```

### 3. Full single-GPU run

Copy a debug config, point `data.tasks` at a larger built distribution, raise
`online.cycles` / `rollout.group_size`, and (on CUDA) switch `model.dtype` to
`bf16`. The Transformers launch templates under `configs/experiments/` are
starting points. Pin exact model/tokenizer revisions before treating any run as
reproducible.

## Inspect, track, and audit a run

```bash
# Learning curves, baseline-vs-final, per-task tables → portable HTML
python3 scripts/render_evaluation_dashboard.py \
  --curve grpo=artifacts/grpo_debug_local \
  --curve ppo=artifacts/ppo_debug_local \
  --output outputs/reports/dashboard.html

# Push metrics to W&B / TensorBoard / JSONL (auto-fallback to JSONL)
python3 scripts/track_run.py --artifact-dir artifacts/grpo_debug_local --backend wandb

# Audit rollouts for reward hacking / degenerate outputs
python3 scripts/scan_reward_hacking.py \
  --rollouts artifacts/grpo_debug_local/cycle_0015/rollouts.jsonl \
  --tasks data/processed/debug_tasks.jsonl

# Paired baseline-vs-candidate statistics
python3 scripts/compare_rollouts.py \
  --baseline outputs/rollouts/baseline.jsonl \
  --candidate outputs/rollouts/candidate.jsonl \
  --output outputs/reports/comparison.md
```

## How to add a dataset

1. Provide tasks in MBPP, HumanEval, or canonical CodeSelf JSONL format
   (loaders in `src/codeself/datasets/loaders.py`).
2. Declare a versioned dataset config (`configs/datasets/*.dataset.yaml`) listing
   the sources, split fractions, seed, and `visible_tests_per_task` for
   auto hidden/visible splitting.
3. `python3 scripts/build_dataset.py --config <cfg> --output <jsonl> --manifest <json>`.
   The build **fails hard** on hidden-test leakage or train/eval contamination
   (override with `--allow-contamination` only for debugging).

## How to add a reward

Implement the `RewardScorer` protocol (`score(result, *, solution_code) ->
RewardBreakdown`) in `src/codeself/rewards/`, or configure the built-in
`ConfigurableRewardScorer` modes: `binary_all_tests_pass`, `fractional_pass_rate`,
`partial_credit`, with optional shaping (compile bonus, length penalty) logged as
separate components. Reward-hacking flags are attached to rollout metadata
automatically.

## Compute requirements

- **Smoke/CI**: any machine, no GPU, seconds.
- **Debug run**: ~8 GB accelerator memory; the 0.6B LoRA GRPO debug run finishes
  in a few minutes per a handful of cycles on an Apple M-series laptop (MPS, fp16).
  Forward/backward are **microbatched** (`training.microbatch_size`) so peak
  memory scales with the microbatch, not the full group.
- **Full run**: a single 16–24 GB GPU for a 0.5B–1.5B coder model with LoRA,
  bf16, and gradient checkpointing. Full fine-tuning is a config option
  (`model.full_finetune: true`).

## Reproducibility

```bash
python3 scripts/make_reproducibility_manifest.py \
  --output outputs/reports/reproducibility_manifest.json \
  --markdown-output outputs/reports/reproducibility_manifest.md
```

The manifest records git revision + dirty status, config/dataset/checkpoint
hashes, environment lockfiles, parsed model/tokenizer references, and the
captured **runtime environment** (accelerator, CPU count, exact torch /
transformers / peft / accelerate / numpy versions).

Release-facing docs:
[reproducibility checklist](./docs/reproducibility_checklist.md) ·
[adapter model card](./docs/model_card_adapters.md) ·
[dataset card](./docs/dataset_card_private_tasks.md) ·
[limitations & safety](./docs/limitations_and_safety.md) ·
[overhaul plan](./docs/overhaul_plan.md).

## Results section template

When reporting a run, include:

1. Dataset version + fingerprint (from the build manifest) and split sizes.
2. Model + exact revision, LoRA config, dtype, accelerator (from the manifest).
3. Reward curve and KL-to-reference over cycles (dashboard or `track_run.py`).
4. Baseline-vs-final pass@k on the **held-out** set with paired bootstrap CI and
   a permutation-test p-value + effect size (`compare_rollouts.py`).
5. Pass-rate by difficulty bucket (`evaluate_rollouts.py --group-by
   metadata.difficulty`).
6. Fraction of flagged/degenerate rollouts (`scan_reward_hacking.py`).
7. Known failure cases and the algorithm comparison (GRPO vs PPO).

## Safety

Generated code is untrusted and executed only in the restricted sandbox
(process jail, CPU/memory/file rlimits, parent-side memory watchdog, no network,
static + runtime hardening, harness isolation so candidates cannot read expected
outputs). Keep hidden tests out of prompts and public traces; reserve locked
private test sets for final evaluation, not checkpoint selection. See
[docs/limitations_and_safety.md](./docs/limitations_and_safety.md).

## License

MIT. See [LICENSE](./LICENSE).
