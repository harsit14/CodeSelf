# Milestone 14: Making Sandbox Results More Useful

The first real implementation step after the overhaul kickoff was not a neural
network loss. It was the executor.

That is not because the executor is more exciting than GRPO. It is because, in
execution-feedback RL, the executor is part of the reward function. If the
sandbox gives vague results, the reward is vague. If it silently conflates
assertion failures, timeouts, and harness issues, then the trainer will optimize
against a blurry signal.

## The problem with phase-level results

The original CodeSelf runner had a nice simple shape:

1. parse the generated Python;
2. scan for obviously unsafe code;
3. import the candidate;
4. run public tests;
5. optionally run hidden tests.

Each of those stages returned a phase result. That was enough for the smoke
pipeline, but it was too coarse for real research. Suppose a task has ten hidden
tests and a candidate passes seven. The old runner could tell me only that the
hidden-test phase failed. For binary pass/fail eval that is acceptable, but for
reward design, debugging, and dashboarding it leaves a lot on the table.

So this milestone adds per-test outcomes while preserving the old phase-level
contract.

## What changed

`PhaseResult` now has a `test_outcomes` field. Each outcome records the test
name, status, duration, stdout, stderr, and error message. Public and hidden
test phases still exist, but they are now aggregates over individual test
subprocess runs.

This buys us a few useful things:

- fractional rewards can later use actual per-test pass rates;
- dashboard views can show exactly which tests failed;
- timeout and failure reporting becomes less mysterious;
- hidden tests skipped after public failure are represented explicitly as
  skipped test outcomes.

I also added a small batch execution helper, `SandboxedTestRunner.run_many()`.
It uses a bounded worker pool and returns results in input order. This is not
yet a production-scale rollout executor, but it gives the next training phase a
clean place to ask for many candidate programs to be evaluated concurrently.

## Timeout cleanup

The subprocess runner now starts POSIX children in a new session and kills the
process group on timeout. This matters for generated code because the annoying
case is not just an infinite loop in the main process. It is a candidate that
spawns work and then leaves descendants behind.

This is still not a perfect jail. The local runner is a fast development
sandbox, not a replacement for container isolation. But terminating the process
group is a meaningful improvement over killing only the direct child.

## What is still unsolved

The honest answer: plenty.

Static scanning is still not a proof of safety. Python is very introspective,
and clever object tricks can bypass many simple AST blocklists. The temporary
run directory still contains a generated harness file. macOS also does not give
the same memory-limit behavior as a Linux cgroup setup, which matters because my
local development machine is a MacBook M5 Pro.

The next sandbox iterations should add adversarial tests for memory pressure,
filesystem access, network attempts, process spawning, and harness
introspection. The Docker final-evaluation path also needs to become executable
end to end instead of remaining mostly a command contract.

## Why this is the right order

This milestone does not make the model smarter. It makes the feedback sharper.
That is the right trade for this stage of the project. Once GRPO starts updating
weights, I want to know that a reward delta is coming from candidate behavior,
not from an executor that hid the interesting details.
