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

Status: pending

Goals:

- Add versioned dataset config loading.
- Generalize custom JSONL ingestion and hidden/visible test auto-splitting.
- Add exact and near-duplicate contamination checks.
- Introduce a reward plugin interface with binary, fractional, and partial
  credit rewards.
- Log shaping terms and reward-hacking flags separately.

## Phase 4: Common RL Training Core

Status: pending

Goals:

- Add model/tokenizer loading with Transformers and PEFT/LoRA.
- Add response masks, prompt masks, token logprobs, old logprobs, reference
  logprobs, entropy, response lengths, and KL metrics.
- Keep trainer backends pluggable so `from_scratch`, TRL, or verl can be chosen
  by config.
- Keep `--smoke` as a fast dependency-free CI path.

## Phase 5: Real GRPO

Status: pending

Goals:

- Implement group sampling per prompt.
- Normalize group-relative advantages.
- Use clipped policy-gradient loss over response tokens only.
- Penalize or constrain KL against a frozen reference model.
- Periodically evaluate on a dev slice.
- Write checkpoints, metrics, rollouts, and reproducibility metadata.

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
