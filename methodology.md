# Methodology

This document is the public research methodology for CodeSelf, a project that
studies whether a coding agent can improve through execution feedback and
reinforcement learning.

## 1. Abstract

CodeSelf investigates bounded self-improvement for Python coding tasks. The
system samples code solutions, executes them in a restricted environment, scores
them with a correctness-first reward, and prepares GRPO-first training and PPO
comparison pipelines. The current repository contains a dependency-free smoke
implementation of the full research workflow: task loading, sandbox execution,
reward scoring, rollout records, baseline evaluation, GRPO diagnostics, PPO
diagnostics, paired statistical testing, agentic traces, and reproducibility
metadata. Real model fine-tuning is the next implementation layer and should
reuse these contracts.

## 2. Research Scope

The project is intentionally narrow. It does not claim open-ended recursive
self-improvement. The goal is to measure whether a base coding model can improve
on a predeclared distribution of programming tasks after practicing with
execution feedback.

Primary research question:

- Does execution-feedback RL improve held-out pass@1 over the base model under a
  fixed compute and rollout budget?

Secondary questions:

- Does GRPO make better use of sparse test-based rewards than PPO for this
  setting?
- How much within-prompt reward variance exists before training?
- Do agentic public-test revisions produce measurable final-test improvement?
- Are observed gains statistically meaningful under paired evaluation?

## 3. System Overview

The CodeSelf pipeline has six stages:

1. Convert benchmark or private tasks into canonical JSONL.
2. Render frozen prompts and sample multiple completions per task.
3. Parse generated code and execute it in a sandboxed runner.
4. Score execution results with `reward_v0_correctness`.
5. Train or smoke-test GRPO/PPO diagnostics from grouped rollout feedback.
6. Evaluate base and candidate systems on held-out tasks with paired statistics.

Every stage writes structured artifacts so later claims can be audited. Rollouts
are JSONL records. Rewards include component breakdowns. Training smoke runs
write metric JSONL and checkpoint manifests. Final evaluation writes Markdown
and JSON reports.

## 4. Datasets And Splits

Supported loaders convert MBPP-style records, HumanEval/EvalPlus-style records,
and canonical CodeSelf JSONL into `TaskSpec` objects. Each task includes a prompt,
entry point, public tests, optional hidden tests, resource limits, tags, and
metadata.

Required splits:

- `train`: used for training rollouts.
- `dev`: used for checkpoint selection and hyperparameter choices.
- `test_public`: publishable held-out evaluation.
- `test_private`: locked final evaluation.
- `stress`: adversarial or distribution-shift probes.

Hidden tests are never placed in prompts or starter code. Split manifests and
dataset fingerprints must be recorded before training begins.

## 5. Model Selection And Headroom Audit

The base model should be chosen only after a headroom audit. A useful RL target
has enough capability to solve some tasks, enough failures to improve, and enough
within-group variance for policy optimization.

The audit should report:

- Parseable-output rate.
- Greedy pass@1.
- Sampled pass@1 and pass@k.
- Per-prompt group success counts.
- Informative-prompt fraction.
- Ceiling risk on dev and private probes.
- Whether supervised warm-start is needed before RL.

If the base model has all-fail groups, there is little learning signal. If it has
all-pass groups, the benchmark is too easy. The current smoke trainers measure
the informative-prompt fraction so this issue is visible early.

## 6. Prompting And Rollouts

Prompt templates are versioned and frozen per experiment. The first templates
ask for a direct Python solution. One template hides public tests, and another
includes public tests for experiments that explicitly test public feedback.
Hidden tests remain excluded in both cases.

Rollout records include:

- Task ID and sample index.
- Prompt template name and rendered prompt.
- Raw completion and parsed code.
- Parser status.
- Generation backend metadata.
- Execution result by phase.
- Reward breakdown and diagnostics.

The smoke repository supports mock, static, and optional local Transformers
generation. Real training should add model revision, tokenizer revision,
sampling parameters, log probabilities, and token counts.

## 7. Execution Environment

Generated code is untrusted. The local runner performs a conservative AST
security scan and executes code in a temporary subprocess with resource limits.
The task runner records syntax, import, public-test, and hidden-test phases.

Training smoke tests can use the local runner for speed. Final evaluation should
use stricter clean isolation, such as the included Docker executor scaffold, and
should record the image digest.

## 8. Reward Function

The MVP reward is `reward_v0_correctness`. It gives primary weight to:

- Syntax success.
- Import success.
- Public-test success.
- Hidden-test success.
- Robustness-test success, when available.
- Safety penalties.

The reward returns a structured breakdown with component weights, normalized
available weights, penalties, and diagnostics. Quality and efficiency are logged
as diagnostics in the MVP and are not optimized directly. This is a deliberate
choice to avoid rewarding style proxies before correctness is reliable.

## 9. GRPO Training Design

GRPO is the primary RL method for the project because code tasks naturally allow
multiple completions for the same prompt. A group of completions can be scored
with tests, and each completion can be compared against the group outcome without
requiring a separate value model.

Metrics to track:

- Mean reward.
- Reward standard deviation.
- Mean absolute group advantage.
- Within-group reward variance.
- Informative-prompt fraction.
- KL, entropy, and token-level diagnostics once real model training is added.
- Execution pass rate and parser failure rate.

The current GRPO implementation is a smoke trainer. It computes grouped
advantages and writes metrics/checkpoint manifests, but it does not update model
weights.

## 10. PPO Baseline

PPO is the comparison baseline. A fair PPO run must match the GRPO rollout
budget and use the same task splits, prompts, reward version, and evaluation
protocol.

Real PPO requires:

- Model log probabilities.
- A value head or value model.
- Advantage estimation.
- Clipped policy objective.
- KL tracking.
- Value loss tracking.
- Actual checkpoint updates.

The current PPO implementation is a smoke scaffold. It computes value-baseline
advantages, simulated clipping diagnostics, and value-loss metrics so the report
and comparison plumbing are ready before real training.

## 11. Agentic Strategy Analysis

Direct generation alone cannot prove that an agent developed strategies such as
writing tests first. Strategy claims require a traceable tool loop where public
tests, revisions, and final submissions are logged.

The first agentic loop supports:

- Public-test execution.
- Custom-test execution for proposed checks.
- A deterministic smoke repair helper.
- Final hidden-test submission.
- JSONL traces and strategy reports.

Strategy claims should report revision rate, public-test pass rate, final pass
rate, mean tool calls, mean reward, and representative traces.

## 12. Evaluation Protocol

Primary metric:

- Held-out pass@1.

Secondary metrics:

- pass@k.
- Parser failure rate.
- Syntax/import success.
- Test pass fraction.
- Runtime and memory.
- Reward statistics.
- Tool-call counts for agentic runs.
- Token cost and latency when real models are used.

Final comparisons should use paired task outcomes over common task IDs. The
current statistical layer reports exact McNemar/binomial p-values over discordant
pairs and paired bootstrap confidence intervals for pass@1 deltas.

Checkpoint selection must use train/dev metrics only. Private test results are
reserved for final locked evaluation.

## 13. Reproducibility Package

The public reproducibility package includes:

- This methodology document.
- `docs/reproducibility_checklist.md`.
- `docs/model_card_adapters.md`.
- `docs/dataset_card_private_tasks.md`.
- `docs/limitations_and_safety.md`.
- A manifest/archive CLI in `scripts/make_reproducibility_manifest.py`.

The manifest records environment metadata, git status, artifact checksums, and
core reproduction commands. Real training runs must extend the manifest with
model revisions, tokenizer revisions, hardware, seeds, checkpoint hashes, raw
rollouts, and final statistical reports.

## 14. Exact Smoke Commands

Validate the example task schema:

```bash
python3 scripts/validate_task_schema.py configs/datasets/tasks.example.jsonl
```

Run the test suite:

```bash
python3 -m unittest discover -s tests
```

Generate mock rollouts:

```bash
python3 scripts/run_rollouts.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/mock_smoke.jsonl \
  --backend mock \
  --samples-per-task 4
```

Evaluate rollouts:

```bash
python3 scripts/evaluate_rollouts.py \
  --rollouts outputs/rollouts/mock_smoke.jsonl \
  --output outputs/reports/mock_baseline.md \
  --power-output outputs/reports/mock_power.json
```

Run GRPO smoke diagnostics:

```bash
python3 scripts/train_grpo_smoke.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output-dir outputs/checkpoints/grpo_smoke_example \
  --backend mock \
  --group-size 4 \
  --max-steps 3
```

Run PPO smoke diagnostics:

```bash
python3 scripts/train_ppo_smoke.py \
  --tasks configs/datasets/tasks.example.jsonl \
  --output-dir outputs/checkpoints/ppo_smoke_example \
  --backend mock \
  --samples-per-task 4 \
  --max-steps 3
```

Compare PPO and GRPO smoke metrics:

```bash
python3 scripts/compare_training_smoke.py \
  --grpo-metrics outputs/checkpoints/grpo_smoke_example/metrics.jsonl \
  --ppo-metrics outputs/checkpoints/ppo_smoke_example/metrics.jsonl \
  --output outputs/reports/ppo_vs_grpo_report.md
```

Create the reproducibility manifest and archive:

```bash
python3 scripts/make_reproducibility_manifest.py \
  --output outputs/reports/reproducibility_manifest.json \
  --markdown-output outputs/reports/reproducibility_manifest.md \
  --archive outputs/reports/codeself_reproducibility.tar.gz
```

## 15. Limitations

The current repository does not yet perform real model fine-tuning. GRPO and PPO
metrics are smoke diagnostics over generated rollouts. The mock backend is useful
for testing code paths, not measuring model skill.

Other limitations:

- Benchmarks may be contaminated by pretraining data.
- Small held-out sets may be underpowered.
- Public tests can be overfit through repeated revision.
- Reward shaping can incentivize shortcuts.
- LoRA capacity may limit improvement.
- RL can regress general coding ability while improving narrow benchmark score.

## 16. Safety And Ethics

Generated code must be sandboxed. Hidden tests and private datasets must not be
published without license and privacy review. Results should include failures and
regressions, not only successful examples. The project should avoid claims of
general autonomous self-improvement unless future evidence supports them.

## 17. Publication Criteria

A final public result should include:

- Frozen model and tokenizer revisions.
- Frozen dataset split manifests.
- Prompt template hashes.
- Reward config hashes.
- Training configs and seeds.
- Checkpoint or adapter checksums.
- Raw rollout files.
- Paired statistical reports.
- Reproducibility manifest.
- Model card and dataset card.
- Limitations and safety statement.
