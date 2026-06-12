# CodeSelf Overhaul Plan

Updated: 2026-06-12

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

Status: complete

Goals:

- Implement group sampling per prompt. Done.
- Normalize group-relative advantages. Done.
- Use clipped policy-gradient loss over response tokens only. Done.
- Penalize or constrain KL against a frozen reference model. Done.
- Periodically evaluate on a dev slice. Done.
- Write checkpoints, metrics, rollouts, and reproducibility metadata. Done.

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

Third slice shipped:

- Added a lazy optional Torch GRPO optimizer-step helper over prepared tensor
  batches.
- Added `GRPOOptimizerStepConfig` for loss config, gradient accumulation,
  gradient clipping, zero-grad behavior, and optimizer stepping.
- Added `GRPOOptimizerStepResult` with detached optimizer-step diagnostics.
- `run_grpo_optimizer_step()` now computes the tensor GRPO loss, scales it for
  gradient accumulation, calls backward, optionally clips gradients, optionally
  steps the optimizer, and returns metrics.
- Added tests for config validation, missing-Torch behavior, and real optimizer
  parameter updates when Torch is installed.

Fourth slice shipped:

- Added a minimal GRPO training loop over prepared tensor batches.
- Added `GRPOTrainingLoopConfig`, `GRPOTrainingLoopStep`, and
  `GRPOTrainingLoopResult` records for microbatch and optimizer-step metrics.
- Added `run_grpo_training_loop()` to select batches, group them for gradient
  accumulation, build differentiable tensor batches, call the optimizer-step
  helper, optionally step a scheduler, and return aggregate metrics.
- Added tests for active Torch gradient accumulation, final partial
  accumulation groups, scheduler stepping, `max_batches`, and empty input
  validation.
- Added `numpy` to the optional training extra after the local Torch-enabled
  validation environment warned without it.
- Validation now runs the Torch GRPO tests in a local `.venv` without
  dependency-related skips.

Fifth slice shipped:

- Added a model-facing GRPO tensor batch builder for Torch causal-LM modules.
- Added `PaddedTrainingTensors` and `pad_training_batch_tensors()` to convert
  `TrainingBatch` records into padded `input_ids`, attention masks, response
  masks, advantages, and optional old/reference logprob tensors.
- Added `gather_causal_lm_token_logprobs()` to collect next-token logprobs from
  `[batch, tokens, vocab]` logits while preserving token alignment with the
  packed prompt/response sequence.
- Added `build_grpo_tensor_batch_from_model()` to run policy, optional
  old-policy, and optional reference forward passes and return a differentiable
  `GRPOTensorBatch`.
- Added active Torch tests with a tiny toy causal-LM parameter table, including
  a test that drives the existing GRPO training loop through the model-facing
  tensor batch builder.

Sixth slice shipped:

- Added a minimal model-aware GRPO trainer over Torch causal-LM modules.
- Added `GRPOModelTrainingConfig` and `GRPOModelTrainingResult` to bind loss,
  optimizer, batch limit, scheduler, dtype, device, and pad-token settings.
- Added `run_grpo_model_training()` to configure model train/eval modes, create
  an AdamW optimizer when one is not supplied, build model-forward tensor
  batches, run the GRPO training loop, and return checkpoint-ready metadata.
- Added optional JSONL metric writing and checkpoint manifest writing for the
  model-training path.
- Added active Torch tests for optimizer creation, parameter updates, metrics
  files, checkpoint manifests, reference-model KL wiring, and invalid config
  cases.

Seventh slice shipped:

- Added explicit Torch state checkpoint writing for the model-aware GRPO
  trainer.
- Added `GRPOCheckpointArtifact` records for checkpoint artifact kind, path,
  byte size, and SHA-256 digest.
- `run_grpo_model_training()` now accepts `state_dir` and can write
  `policy_model.pt`, `optimizer.pt`, and optional `scheduler.pt` state files.
- Checkpoint manifests now include `has_state_artifacts` and artifact checksum
  metadata when state files are written.
- Active Torch tests now load the saved policy state, verify optimizer state
  artifacts, and check SHA-256 metadata.

Eighth slice shipped:

- Added a rollout-to-GRPO batch bridge that converts `RolloutRecord` objects
  into tokenized `TrainingBatch` samples.
- The bridge uses raw model completions by default, with an explicit
  `parsed_code` option for experiments that should optimize only extracted
  Python code.
- Added parse-failure, empty-response, and invalid-reward skip diagnostics so
  bad rollout streams fail visibly instead of silently producing weak batches.
- Group-relative advantages are assigned as part of batch construction, keeping
  the output ready for the existing GRPO tensor/model trainer path.
- Added metadata carry-through for backend, model name, parse status, reward
  name, execution pass/fail, truncation, and scalar rollout/generation fields.

Ninth slice shipped:

- Added `run_grpo_rollout_training_cycle()` as the first collect -> batch ->
  optimize orchestration layer for real GRPO.
- The cycle calls the existing rollout generator, optionally writes rollout
  JSONL, converts rollouts into a GRPO `TrainingBatch`, then runs the
  model-aware GRPO trainer.
- Added `GRPORolloutTrainingCycleConfig` to bind sampling settings, rollout
  batch settings, hidden-test inclusion, seed, and model-training settings in
  one object.
- The cycle defaults to `kl_beta=0.0` so it can run without a reference model;
  callers can still pass a reference model and nonzero KL through the existing
  training config.
- Added a Torch-backed unit test that runs the full cycle with a tiny model,
  writes rollout/metric/checkpoint artifacts, and verifies a parameter update.

Tenth slice shipped:

- Added `run_grpo_online_training()` as a repeated collect -> execute -> score
  -> optimize loop over multiple GRPO rollout-training cycles.
- Added `GRPOOnlineTrainingConfig`, per-cycle seed progression, per-cycle result
  summaries, and aggregate rollout/optimizer-step/reward diagnostics.
- The online runner creates one optimizer when the caller does not provide one,
  preserving optimizer state across cycles instead of rebuilding AdamW each
  cycle.
- Added cycle-scoped artifact directories with rollout JSONL, metrics,
  checkpoint manifests, and Torch state files for each online cycle.
- Added a Torch-backed unit test that runs two full cycles and verifies artifact
  layout, seed progression, optimizer steps, and parameter updates.

Eleventh slice shipped:

- Added old-policy snapshot syncing for repeated GRPO online cycles.
- `GRPOOnlineTrainingConfig` now defaults to syncing a supplied old-policy
  model from the current policy before each cycle.
- Each `GRPOOnlineTrainingStep` records whether the old-policy snapshot was
  synced, making the ratio baseline visible in serialized diagnostics.
- Added validation that the policy exposes `state_dict()` and the old-policy
  model exposes `load_state_dict()` when snapshot syncing is enabled.
- Extended the Torch-backed online test to verify one old-policy load per cycle.

Twelfth slice shipped:

- Added optional dev-set evaluation cadence to `run_grpo_online_training()`.
- Online GRPO can now generate evaluation rollouts after each cycle, write
  `eval_rollouts.jsonl`, write `evaluation.json`, and attach the evaluation
  summary to the per-cycle result.
- Added `ModelEngineCodeGenerator` so a shared training `ModelEngine` can drive
  rollout generation while exposing the same tokenizer/model pair for GRPO.
- Exposed the underlying model and raw tokenizer from `TransformersModelEngine`
  for advanced from-scratch GRPO integration.
- Extended checkpoint writing to preserve `save_pretrained()` artifacts, which
  covers PEFT/LoRA-style adapter checkpoints without importing PEFT in tests.
- Added tests for dev evaluation artifacts, model-engine generation, and
  `save_pretrained()` checkpoint artifacts.

Phase 5 completion note:

- Real GRPO now has dependency-free reference math, Torch tensor losses,
  optimizer steps, model-forward batches, model-aware training, state and
  adapter-style checkpoints, rollout-to-batch conversion, one-cycle training,
  repeated online cycles, old-policy snapshot syncing, dev evaluation cadence,
  and a shared-policy generator adapter.
- Remaining work now belongs to later named phases: PPO parity, richer
  evaluation/statistics/dashboarding, self-debug rollout mode, and final
  paper-ready packaging.

## Phase 6: PPO Baseline

Status: complete

Goals:

- Add a value head. Started with value-estimate records.
- Implement GAE, clipped policy loss, clipped value loss, entropy, and KL. Started.
- Match rollout budgets and evaluation protocol with GRPO.

First slice shipped:

- Added dependency-free PPO value-target and loss helpers.
- Added `PPOLossConfig` with policy clipping, value clipping, value-loss weight,
  entropy bonus, optional reference KL, gamma, GAE lambda, and advantage
  normalization settings.
- Added `PPOValueEstimate`, `PPOValueTarget`, and `PPOValueTargetResult` records
  to compute sparse-final-reward GAE over response tokens.
- Added `compute_ppo_loss()` with clipped policy loss, clipped value loss,
  entropy loss, and optional reference-model KL over response tokens only.
- Added tests for GAE targets, policy/value clipping, entropy, reference KL,
  advantage normalization, and validation failures.

Second slice shipped:

- Added an optional Torch implementation of the PPO objective behind a lazy
  import boundary.
- Added `PPOTensorBatch` and `PPOTensorLossResult` records for differentiable
  policy, value, entropy, and KL loss computation.
- Added `build_ppo_tensor_batch()` to convert `TrainingBatch` records and PPO
  value estimates into padded policy, old-policy, response-mask, advantage,
  return, value, old-value, entropy, and optional reference tensors.
- Added `compute_ppo_tensor_loss()` with the same response-token masking,
  policy clipping, value clipping, entropy bonus, and reference-KL semantics as
  the dependency-free reference implementation.
- Added active Torch tests for numerical parity with the reference objective,
  variable-length padding, and differentiability through policy and value terms.

Third slice shipped:

- Added a lazy optional Torch PPO optimizer-step helper over prepared tensor
  batches.
- Added `PPOOptimizerStepConfig` for loss config, gradient accumulation,
  gradient clipping, zero-grad behavior, and optimizer stepping.
- Added `PPOOptimizerStepResult` with detached diagnostics for total, policy,
  value, entropy, KL, ratio, advantage, return, and clipping metrics.
- `run_ppo_optimizer_step()` now computes the tensor PPO loss, scales it for
  gradient accumulation, calls backward, optionally clips gradients, optionally
  steps the optimizer, and returns metrics.
- Added active Torch tests for config validation, missing-Torch behavior, real
  policy/value parameter updates, and accumulation without optimizer stepping.

Fourth slice shipped:

- Added model-facing PPO tensor batch helpers for Torch causal-LM modules with
  value heads.
- Added `PaddedPPOTrainingTensors` and `pad_ppo_training_batch_tensors()` to
  pad PPO samples without requiring GRPO-style group advantages.
- Added `gather_causal_lm_token_entropy()` to compute next-token categorical
  entropy aligned to response-token positions.
- Added `build_ppo_tensor_batch_from_model()` to run policy, optional value,
  old-policy, old-value, and reference forward passes into a differentiable
  `PPOTensorBatch`.
- The PPO model bridge keeps old policy logprobs, old values, and reference
  logprobs detached while leaving policy logprobs, entropy, and current value
  predictions differentiable.
- Added active Torch tests for entropy alignment, recorded rollout logprobs,
  value-head gradients, optimizer updates, detached old/reference models,
  explicit rollout-time value estimates, and missing value-head failures.

Fifth slice shipped:

- Added a minimal PPO training loop over prepared tensor batches.
- Added `PPOTrainingLoopConfig`, `PPOTrainingLoopStep`, and
  `PPOTrainingLoopResult` records for microbatch and optimizer-step metrics.
- Added `run_ppo_training_loop()` to select batches, group them for gradient
  accumulation, build differentiable tensor batches, call the PPO optimizer-step
  helper, optionally step a scheduler, and return aggregate diagnostics.
- Added active Torch tests for gradient accumulation, final partial
  accumulation groups, scheduler stepping, `max_batches`, value-loss metrics,
  and empty input validation.

Sixth slice shipped:

- Added a minimal model-aware PPO trainer over Torch causal-LM modules and value
  heads.
- Added `PPOModelTrainingConfig`, `PPOModelTrainingResult`,
  `PPOCheckpointArtifact`, and `PPOValueEstimateProvider` records.
- Added `run_ppo_model_training()` to configure model modes, create an AdamW
  optimizer when one is not supplied, build model-forward PPO tensor batches,
  run the PPO training loop, and return checkpoint-ready metadata.
- The trainer supports integrated value heads, separate value models,
  old-policy snapshots, old-value models, reference models, callable or static
  rollout-time value estimates, JSONL metric writing, checkpoint manifests, and
  Torch state artifacts.
- Added active Torch tests for integrated policy/value updates, separate value
  model state, old/reference model evaluation modes, adapter-style
  `save_pretrained()` artifacts, config validation, and missing trainable
  parameter failures.

Seventh slice shipped:

- Added a rollout-to-PPO batch bridge that converts `RolloutRecord` objects into
  tokenized `TrainingBatch` samples without assigning GRPO-style group
  advantages.
- Added `PPORolloutBatchConfig` and `PPORolloutBatchResult` with the same
  prompt/response token limits, response-source selection, parse-failure
  handling, skip diagnostics, and metadata carry-through used by the GRPO
  bridge.
- Added `run_ppo_rollout_training_cycle()` as the first collect -> batch ->
  optimize orchestration layer for PPO.
- Added `PPORolloutTrainingCycleConfig` and
  `PPORolloutTrainingCycleResult` to bind sampling settings, rollout batch
  settings, hidden-test inclusion, seed, and model-training settings in one
  object.
- The cycle writes optional rollout JSONL, metric JSONL, checkpoint manifest,
  and Torch state artifacts while calling the model-aware PPO trainer.
- Added dependency-free rollout-batch tests and a Torch-backed cycle test that
  collects rollouts, trains the integrated policy/value toy model, writes
  artifacts, and verifies policy and value updates.

Eighth slice shipped:

- Added `run_ppo_online_training()` as a repeated collect -> execute -> score
  -> optimize loop over multiple PPO rollout-training cycles.
- Added `PPOOnlineTrainingConfig`, `PPOOnlineTrainingStep`, and
  `PPOOnlineTrainingResult` with per-cycle seed progression, aggregate rollout
  counts, records-used counts, optimizer-step counts, and weighted mean reward.
- Added `PPOOnlineEvaluationConfig` and `PPOOnlineEvaluationResult` so PPO can
  run dev-set evaluation rollouts after each cycle and write
  `eval_rollouts.jsonl` plus `evaluation.json`.
- The online runner creates one optimizer when the caller does not provide one,
  preserving optimizer state across cycles instead of rebuilding AdamW each
  cycle.
- Added old-policy and old-value snapshot syncing before each cycle, with
  explicit per-cycle diagnostics for whether each snapshot was synced.
- Added cycle-scoped artifact directories with rollout JSONL, metrics,
  checkpoint manifests, Torch state files, evaluation rollouts, and evaluation
  reports.
- Added Torch-backed tests for seed progression, config validation, old-policy
  and old-value sync, repeated optimizer updates, artifact layout, dev
  evaluation cadence, and policy/value parameter updates.

Phase 6 completion note:

- PPO now has dependency-free reference math, Torch tensor losses, optimizer
  steps, model/value-head tensor batches, a tensor training loop, model-aware
  training, state and adapter-style checkpoints, rollout-to-batch conversion,
  one-cycle training, repeated online cycles, old-policy and old-value snapshot
  syncing, and dev evaluation cadence.
- Remaining work now belongs to later named phases: richer
  evaluation/statistics/dashboarding, self-debug rollout mode, and final
  paper-ready packaging.

## Phase 7: Evaluation, Statistics, And Dashboard

Status: started

Goals:

- Expand pass@k reporting and sample-count controls.
- Add paired permutation tests and effect sizes with confidence intervals.
- Plot reward, KL, entropy, length, pass rate by difficulty, and degenerate
  output fraction.
- Build a local dashboard or plotting scripts for learning curves, win/loss
  tables, and rollout browsing.

First slice shipped:

- Expanded `evaluate_rollouts()` with an optional `max_samples_per_task` control
  so reports can compare fixed rollout budgets without rewriting JSONL files.
- Added per-task pass@k dictionaries, degenerate-output counts/rates, and
  response-token means to `TaskEvaluation`.
- Added aggregate reward standard deviation, degenerate-output rate, response
  token mean/median/max, and dashboard-ready `GroupEvaluation` records to
  `EvaluationSummary`.
- Added metadata/field-based group summaries through `group_by`, supporting
  fields such as `metadata.difficulty`, `metadata.split`, or `reward.reward_name`.
- Updated `scripts/evaluate_rollouts.py` with `--max-samples-per-task` and
  `--group-by`, plus console output for degenerate rate, response length, and
  group pass@1.
- Added tests for capped sample budgets, group summaries, per-task pass@k,
  response-token statistics, degenerate-output detection, and CLI reporting.

Second slice shipped:

- Added `PairedPermutationResult` and `paired_permutation_test()` for paired
  sign-flip permutation testing over task-level pass/fail deltas.
- The permutation test enumerates all sign flips exactly for small discordant
  task sets and uses seeded Monte Carlo sampling for larger comparisons.
- Added `EffectSizeSummary` and `effect_size_summary()` with pass-rate delta,
  percentage-point delta, relative pass-rate lift, base/candidate error rates,
  relative error reduction, reward delta, paired reward-delta standard
  deviation, and standardized reward delta.
- `ComparisonSummary` now includes exact McNemar, paired permutation,
  bootstrap CI, and effect-size payloads in both JSON and Markdown reports.
- Updated `scripts/compare_rollouts.py` with `--permutation-samples` and
  console output for permutation p-values and effect sizes.
- Added tests for exact permutation behavior, effect-size calculations, report
  serialization, CLI output, and validation errors.

Third slice shipped:

- Added a dependency-free static HTML dashboard renderer for evaluation and
  paired-comparison JSON reports.
- Added `render_evaluation_dashboard()`, `write_evaluation_dashboard()`, and
  `read_dashboard_json()` for programmatic dashboard generation.
- Added `scripts/render_evaluation_dashboard.py` with repeated `--evaluation`
  and `--comparison` inputs, labeled report paths, custom titles, and portable
  HTML output.
- The dashboard includes evaluation summary tables, inline SVG pass@1 and delta
  charts, grouped metrics, per-task browsing tables, paired-comparison tables,
  bootstrap CI columns, permutation p-values, and task-flip tables.
- Added tests for Python rendering, HTML escaping, evaluation/comparison
  sections, inline charts, CLI generation, and report counts.

Fourth slice shipped:

- Added `LearningCurvePoint` and `LearningCurveRun` records for summarizing
  online GRPO/PPO artifact directories over repeated training cycles.
- Added `collect_learning_curve()` and `collect_learning_curves()` to parse
  `cycle_*` directories containing rollout JSONL, metric JSONL, checkpoint
  manifests, and optional evaluation reports.
- The collector reports cycle counts, rollout counts, optimizer steps, train
  pass@1, train reward, response length, degenerate-output rate, train losses,
  KL terms, and dev-evaluation metrics when available.
- Extended `render_evaluation_dashboard()`, `write_evaluation_dashboard()`,
  and `scripts/render_evaluation_dashboard.py` with labeled `--curve`
  artifact inputs.
- The dashboard now includes a Learning Curves section with cycle tables and
  inline SVG trend charts for train/eval pass@1, reward, loss, entropy, KL,
  response length, and degenerate-output rate.
- Added tests for artifact parsing, metric aggregation, missing-directory
  validation, dashboard rendering, CLI curve inputs, and report counts.

Remaining risks:

- Degenerate-output detection is intentionally heuristic; it flags empty outputs
  and obvious repetition, but richer task-aware failure tagging still belongs in
  later dashboard work.
- Learning-curve plotting now reads multi-cycle artifact directories, but richer
  raw rollout browsing and filterable failure taxonomies still belong in later
  dashboard work.
- KL and entropy curves depend on trainer metric availability. Missing fields
  currently render as zero rather than trying to infer absent optimization
  diagnostics.

## Phase 8: Self-Debug Training Mode

Status: started

Goals:

- Make generate -> execute -> feedback -> revise a first-class rollout mode.
  Started.
- Reward final attempts, with optional discounted credit across rounds. Started.
- Add single-shot versus self-debug ablation configs. Started.

First slice shipped:

- Added `SelfDebugRolloutConfig` and `SelfDebugRolloutResult` as the bridge
  between agent traces and training-ready rollout records.
- Added `generate_self_debug_rollouts()` to run the existing public-test
  self-debug loop over task/sample grids while returning ordinary
  `RolloutRecord` objects plus trace artifacts.
- Added `rollout_record_from_trace()` so the final revised attempt becomes the
  rollout response, while trace metadata records revision counts, tool calls,
  public-test pass status, final pass status, and initial parse status.
- `AgentToolbox` now accepts injected reward scorers, and final submissions can
  respect `include_hidden`, so self-debug rollouts share reward modes and
  public-only evaluation behavior with direct rollouts.
- Added optional positive-reward discounting by revision count through
  `revision_reward_discount`, preserving the undiscounted final reward in
  reward metrics.
- Extended `scripts/run_rollouts.py` with `--rollout-mode self_debug`,
  `--trace-output`, `--max-revisions`, `--disable-rule-repair`, and
  `--revision-reward-discount`.
- Added single-shot and self-debug ablation config examples under
  `configs/experiments/`.
- Added tests for trace-to-rollout conversion, discounted final rewards,
  self-debug metadata, CLI trace output, and downstream GRPO/PPO rollout-cycle
  compatibility.

Remaining risks:

- The revision policy is still rule-based for smoke testing. A model-generated
  revision prompt that includes structured feedback is the next substantive
  self-debug training slice.
- Current training loops still collect direct rollouts internally. They can
  consume self-debug rollout JSONL through the batch builders, but online GRPO
  and PPO need config switches before self-debug collection is native there.
- Discounting is intentionally simple: positive final reward is scaled by
  `discount ** revision_count`, while non-positive outcomes are left unchanged.

## Phase 9: Documentation And Paper-Ready Package

Status: pending

Goals:

- Rewrite the README around the final architecture.
- Extend reproducibility manifests with config hashes, model/tokenizer
  versions, dataset versions, environment lockfiles, and checkpoint hashes.
- Fill in result templates with plots, confidence intervals, and known failure
  cases.
