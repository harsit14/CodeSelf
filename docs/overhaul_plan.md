# CodeSelf Overhaul Plan

Updated: 2026-06-11

This is the living engineering record for turning CodeSelf from a smoke
research scaffold into a runnable execution-feedback RL framework. Each phase
should end with tests, a short summary, and a commit.

## Local Compute Assumptions

The local development machine is a MacBook M5 Pro with 48GB unified memory.
That is a useful debug box for CPU/MPS smoke runs, dataset tooling, sandbox
hardening, dashboards, and tiny-model integration tests. The main paper-grade
training target should still be written as a pluggable single-GPU path because
bf16 CUDA training on a small coder model is the most portable target for
reproducible RL experiments.

Default assumptions:

- Local smoke/debug: dependency-free path plus optional PyTorch MPS.
- Local tiny-model tests: very small model, low sequence length, LoRA enabled.
- Main single-GPU run: Qwen2.5-Coder 0.5B or 1.5B class model with PEFT/LoRA,
  bf16 on CUDA where available, gradient checkpointing, and small batch sizes.
- Full fine-tuning: config-gated, not the default.

## Phase 1: Smoke Boundary And Documentation

Status: in progress

Goals:

- Keep the dependency-free smoke pipeline green.
- Move GRPO/PPO smoke trainers under an explicit `training/smoke/` namespace.
- Preserve old imports and scripts through compatibility wrappers.
- Add this living plan and a blog-style implementation log.

Verification:

- `python3 -m unittest discover -s tests`
- Existing smoke scripts should continue to import `GRPOSmokeTrainer` and
  `PPOSmokeTrainer` from `codeself.training`.

## Phase 2: Sandbox Hardening

Status: started

Goals:

- Add process-group cleanup and robust timeout handling. Started.
- Run each test with structured per-test outcomes. Done in the first sandbox
  slice.
- Add worker-pool batch execution. Done in the first sandbox slice.
- Strengthen resource limits for CPU, memory, files, and subprocesses where the
  host OS permits. Started with open-file, file-size, core-file, CPU, memory,
  and process rlimits where available.
- Keep final evaluation container-ready with `--network none`, read-only
  mounts, pids limit, memory limit, and non-root execution.
- Add adversarial tests for infinite loops, fork attempts, memory pressure,
  filesystem access, network attempts, `sys.exit`/`os._exit`, and harness
  introspection.

First slice shipped:

- `PhaseResult` now carries `test_outcomes` for per-test pass/fail/timeout
  reporting.
- Public and hidden test phases execute each test snippet in its own subprocess
  phase, then aggregate the phase status for backward compatibility.
- Hidden tests skipped after public failure now include skipped per-test
  outcomes.
- `SandboxedTestRunner.run_many()` runs many candidate programs concurrently
  with a bounded worker pool and returns results in input order.
- Subprocess timeouts now terminate the spawned process group on POSIX hosts.

Second slice shipped:

- Shared phase harness code now catches `SystemExit`, `KeyboardInterrupt`, and
  `GeneratorExit` so generated code cannot turn a control-flow exit into a
  false pass.
- Static scanning rejects direct `SystemExit`, `exit`, `quit`, reflection
  helpers, and common Python introspection escape surfaces such as
  `__class__`, `__subclasses__`, `__globals__`, and `__builtins__`.
- Runtime startup hardening blocks `open`, `input`, `eval`, `breakpoint`,
  `exit`, and `quit` as a defense-in-depth layer for paths that bypass the
  static scan.
- `DockerSandboxRunner.run_phase()` now writes the same phase harness files and
  executes them through Docker, with an injectable command runner so unit tests
  do not require Docker to be installed.
- Docker commands now include no network, read-only root, all capabilities
  dropped, no-new-privileges, pids limit, CPU limit, memory and memory-swap
  limits, non-root user, read-only task mount, and a small `/tmp` tmpfs.

Remaining risks:

- The local subprocess path is still not a full jail. Final evaluation needs the
  container path to be exercised in a slow integration test against a built
  image.
- Static scanning remains bypassable by sufficiently adversarial Python object
  tricks.
- The test harness still materializes executable test code in the temporary
  run directory; future work should reduce harness introspection and leakage
  surfaces.
- macOS does not provide the same memory-enforcement behavior as Linux
  `RLIMIT_AS`, so memory-abuse tests need host-aware expectations.

## Phase 3: Data And Reward Pipeline

Status: started

Goals:

- Add versioned dataset config loading.
- Generalize custom JSONL ingestion and hidden/visible test auto-splitting.
- Add exact and near-duplicate contamination checks. Started.
- Introduce a reward plugin interface with binary, fractional, and partial
  credit rewards. Done in the first Phase 3 slice.
- Log shaping terms and reward-hacking flags separately. Started.

First slice shipped:

- Added dataset quality reporting for hidden-test leakage and train/eval
  contamination.
- Hidden-test leakage now checks prompt, starter code, and public tests.
- Train/eval contamination now detects exact duplicates and normalized
  near-duplicates across split boundaries.
- `scripts/validate_task_schema.py` now reports hidden leaks and contamination
  findings and exits nonzero when either is present.
- Added `ConfigurableRewardScorer` with `binary_all_tests_pass`,
  `fractional_pass_rate`, and `partial_credit` modes.
- Existing `reward_v0_correctness` now uses per-test outcomes when available,
  instead of reducing an entire test phase to one binary value.
- `scripts/audit_reward.py` can audit the new reward modes while retaining
  `correctness_v0` as the default.
- Added `configs/rewards/reward_modes.example.json` to document the first
  pluggable reward choices.

Second slice shipped:

- Added a dependency-free config loader for JSON and the simple YAML subset used
  by repository experiment configs.
- `scripts/run_rollouts.py` now accepts `--config` and can read task path,
  split, generation settings, prompt template, execution settings, reward mode,
  and output path from config.
- CLI values override config values when provided, preserving the old direct
  command style.
- Rollout generation now accepts an injected reward scorer and records
  `reward_mode`/`config_path` metadata on each rollout.
- `run_rollouts.py` runs dataset quality checks before generating rollouts
  unless explicitly disabled.
- `configs/experiments/rollout_smoke.example.yaml` now declares data-quality and
  reward settings and is executable through `scripts/run_rollouts.py --config`.

Third slice shipped:

- GRPO and PPO smoke trainers now accept injected reward scorers.
- `GRPOSmokeConfig` and `PPOSmokeConfig` record the active `reward_mode`.
- `scripts/train_grpo_smoke.py` and `scripts/train_ppo_smoke.py` now accept
  `--config` using the same JSON/simple-YAML loader as rollout generation.
- GRPO/PPO smoke scripts now run dataset quality checks before training unless
  explicitly disabled.
- GRPO/PPO smoke scripts now support reward modes and optional shaping flags.
- Smoke checkpoint manifests now include `reward_mode` and `config_path`.
- `configs/experiments/grpo_smoke.local.example.json` and
  `configs/experiments/ppo_smoke.local.example.json` now declare data-quality,
  execution, and reward settings.

Remaining risks:

- Near-duplicate detection is a simple normalized string similarity check; it
  should be complemented with code-aware or embedding-based review before final
  claims.
- Reward-hacking detection is still shallow. The new reward metrics flag empty
  code and repeated lines, but hardcoded visible-test outputs need task-aware
  analysis.
- Rollout generation and smoke trainers now share config/reward selection, but
  the real training-core skeleton still needs to consume the same config fields.

## Phase 4: Common RL Training Core

Status: started

Goals:

- Add model/tokenizer loading with Transformers and PEFT/LoRA.
- Add response masks, prompt masks, token logprobs, old logprobs, reference
  logprobs, entropy, response lengths, and KL metrics. Started.
- Keep trainer backends pluggable so `from_scratch`, TRL, or verl can be chosen
  by config. Started.
- Keep `--smoke` as a fast dependency-free CI path.

First slice shipped:

- Added dependency-free `codeself.training.common` contracts for future real
  training backends.
- Added backend registry/availability checks for `smoke`, `from_scratch`, TRL,
  and verl.
- Added model runtime, optimizer, rollout runtime, and top-level training-core
  config records.
- Added prompt/response/attention mask helpers that explicitly separate prompt
  tokens from response tokens for loss computation.
- Added token logprob records and response-only summaries for policy logprobs,
  old-policy logprobs, reference logprobs, KL, importance ratios, and entropy.
- Added sequence/batch records for tokenized policy-gradient samples.
- Added `scripts/inspect_training_core.py` to inspect a config and backend
  availability without importing heavyweight training libraries.
- Added `configs/experiments/training_core_debug.example.json` as the first
  real-training-core config example.

Second slice shipped:

- Added dependency-free tokenizer contracts and a deterministic whitespace
  tokenizer for smoke tests.
- Added prompt/response token packing with default left-truncation for prompts
  and right-truncation for responses.
- Added generated-sequence records that can convert into existing
  `SequenceTrainingSample` records.
- Added model-engine and generation-request contracts for future GRPO/PPO
  trainers.
- Added lazy Hugging Face Transformers tokenizer/model adapters. Importing
  `codeself.training` still does not import Torch or Transformers.
- Added token-aligned policy-logprob extraction for the optional Transformers
  engine.
- Added `configs/experiments/model_engine_debug.example.json` for the first
  local model-engine target.

Remaining risks:

- Transformers loading is only an optional engine shell. It still needs a slow
  integration test with an actually cached tiny model.
- PEFT/LoRA adapter attachment is not implemented yet.
- No GRPO/PPO loss is implemented yet; the current work defines the data and
  masking contracts those losses should consume.
- Backend availability checks report optional dependency presence, but do not
  validate CUDA/MPS memory, model licenses, or checkpoint writeability.

## Phase 5: Real GRPO

Status: started

Goals:

- Implement group sampling per prompt.
- Normalize group-relative advantages. Started.
- Use clipped policy-gradient loss over response tokens only. Started.
- Penalize or constrain KL against a frozen reference model. Started.
- Periodically evaluate on a dev slice.
- Write checkpoints, metrics, rollouts, and reproducibility metadata.

First slice shipped:

- Added dependency-free GRPO advantage assignment for `TrainingBatch` records,
  grouped by task/prompt ID.
- Added normalized group-relative advantages with zero-variance groups mapped to
  zero advantage.
- Added a clipped GRPO surrogate loss over response tokens only.
- Added optional reference-model KL penalty using the same token-aligned
  logprob records introduced in Phase 4.
- Added batch and per-sample diagnostics for policy loss, KL loss, total loss,
  mean ratio, clipped ratio, approximate KL, and clipped-token fraction.
- Kept the implementation in plain Python floats so the objective can be tested
  before moving it into Torch tensors.

Second slice shipped:

- Added an optional Torch implementation of the GRPO objective behind a lazy
  import boundary.
- Added `GRPOTensorBatch` and `GRPOTensorLossResult` records for differentiable
  loss computation and detached diagnostics.
- Added `build_grpo_tensor_batch()` to convert `TrainingBatch` records into
  padded policy, old-policy, reference, response-mask, and advantage tensors.
- Added `compute_grpo_tensor_loss()` with the same response-token mask,
  clipping, reference-KL, and per-sequence reduction semantics as the
  dependency-free reference implementation.
- Added import-safe optional-dependency helpers so the smoke test path still
  runs without Torch installed.
- Added tests that verify missing-Torch behavior now and numerical parity with
  the reference implementation when Torch is installed.

Remaining risks:

- This is not yet a full training step and does not update weights.
- Old-policy and reference logprobs still need to be produced by real model
  engines during rollout collection.
- The tensor loss is ready for backpropagation, but no optimizer, scheduler,
  gradient accumulation, checkpointing, or LoRA attachment is wired yet.
- Group sampling is still represented by batch grouping rather than a full
  generate -> execute -> score -> optimize loop.

## Phase 6: PPO Baseline

Status: pending

Goals:

- Add a value head.
- Implement GAE, clipped policy loss, clipped value loss, entropy, and KL.
- Match rollout budgets and evaluation protocol with GRPO.

## Phase 7: Evaluation, Statistics, And Dashboard

Status: pending

Goals:

- Expand pass@k reporting and sample-count controls.
- Add paired permutation tests and effect sizes with confidence intervals.
- Plot reward, KL, entropy, length, pass rate by difficulty, and degenerate
  output fraction.
- Build a local dashboard or plotting scripts for learning curves, win/loss
  tables, and rollout browsing.

## Phase 8: Self-Debug Training Mode

Status: pending

Goals:

- Make generate -> execute -> feedback -> revise a first-class rollout mode.
- Reward final attempts, with optional discounted credit across rounds.
- Add single-shot versus self-debug ablation configs.

## Phase 9: Documentation And Paper-Ready Package

Status: pending

Goals:

- Rewrite the README around the final architecture.
- Extend reproducibility manifests with config hashes, model/tokenizer
  versions, dataset versions, environment lockfiles, and checkpoint hashes.
- Fill in result templates with plots, confidence intervals, and known failure
  cases.
