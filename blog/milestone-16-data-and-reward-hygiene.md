# Milestone 16: Data Hygiene And Reward Modes

This milestone starts the third phase of the overhaul: making the task data and
reward function less implicit.

The project already had a canonical task schema and a correctness-first reward.
That was enough for the smoke pipeline. But once the project starts training a
model, "enough for smoke" is not enough for research. I need explicit checks for
data leakage and explicit reward modes that can be compared in ablations.

## Dataset quality checks

The first addition is a dataset quality report. It currently checks two things:

- hidden-test leakage into prompts, starter code, or public tests;
- train/eval contamination through exact and near-duplicate task text.

The hidden-test check is deliberately boring. It normalizes text and looks for
hidden tests copied into surfaces the model can see. That catches the most
direct mistakes.

The contamination check compares train tasks against dev, public test, private
test, and stress tasks. It reports exact duplicates and near duplicates based on
normalized string similarity. This is not a perfect semantic contamination
detector, but it is a useful first gate. If two tasks differ only by punctuation
or a few words, I want to know before training starts.

I also wired these checks into `scripts/validate_task_schema.py`. That script
used to validate schema shape and hidden-test leakage with a small local check.
Now it calls the shared quality report and exits nonzero if it finds leakage or
train/eval contamination.

## Reward modes

The second addition is a pluggable reward scorer. It keeps the existing
`RewardBreakdown` output format, which matters because rollout logs and
dashboards should not need to special-case every reward experiment.

There are three first modes:

- `binary_all_tests_pass`: sparse reward, 1 only when the full task passes;
- `fractional_pass_rate`: reward is the fraction of non-skipped tests that pass;
- `partial_credit`: syntax, import, public-test, and hidden-test components with
  configurable penalties and optional shaping.

This is where the per-test execution work from the previous milestone starts to
pay off. Fractional rewards can now use real per-test outcomes instead of
pretending an entire phase is one test.

I also updated the old `reward_v0_correctness` logic so its public/hidden test
components use per-test pass fractions when available. That preserves the
existing reward name and output structure, but makes the signal more accurate.

## Small reward-hacking signals

The new configurable scorer logs a couple of simple degenerate-output metrics:
empty code and repeated-line fraction. These are not sufficient reward-hacking
defenses, but they start the pattern I want: detection should be logged
separately from the scalar reward so later analysis can ask whether training is
improving correctness or just finding weird shortcuts.

The harder task-aware checks are still ahead. For example, detecting hardcoded
visible-test outputs requires looking at the task tests and generated code
together. That should become its own module rather than being hidden inside a
scalar reward function.

## What this unlocks

This milestone makes reward ablations much cleaner. Instead of editing the
reward function by hand for every experiment, the framework can choose a reward
mode from config and keep the same logging contract.

It also makes data quality part of the normal workflow. A dataset that leaks
hidden tests or duplicates train tasks into eval should fail early, not after a
model has already learned from it.

The next step is to connect these pieces to experiment configs and rollout
generation, then add stronger task-aware reward-hacking checks.
