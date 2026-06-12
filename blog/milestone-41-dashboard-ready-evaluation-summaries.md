# Milestone 41: Dashboard-Ready Evaluation Summaries

Phase 7 starts with the evaluator.

After closing PPO parity, the project has two trainable algorithm paths that can
produce rollout artifacts. The next problem is not more optimizer plumbing. It
is making the evaluation output rich enough that plots, dashboards, and final
tables do not have to reverse-engineer every metric from raw JSONL.

## What landed

The main changes are in `evaluate.py`.

`evaluate_rollouts` now supports:

- fixed sample budgets with `max_samples_per_task`;
- group summaries through `group_by`;
- per-task pass@k dictionaries;
- aggregate reward standard deviation;
- degenerate-output rates;
- response-token mean, median, and max.

The command-line evaluator also grew:

- `--max-samples-per-task`;
- `--group-by`.

That means I can ask a report to compare only the first `N` samples per task,
or to summarize metrics by fields like `metadata.difficulty`,
`metadata.split`, or `reward.reward_name`.

## Why this matters

Training curves are only useful if the evaluation payload already contains the
right axes. A dashboard should be able to plot pass rate by difficulty, reward
by split, response length by cycle, and degenerate-output fraction without
reimplementing the evaluator.

This slice makes those fields first-class in the JSON report.

## A small heuristic

I added a deliberately simple degenerate-output detector. It catches empty
outputs and obvious repetition. That is not a substitute for a real failure
taxonomy, but it gives the training dashboard an early warning signal. If reward
improves while degenerate-output rate rises, the run deserves skepticism.

## What comes next

The next Phase 7 pieces should add richer paired statistics and visualization:
paired permutation tests, effect sizes, confidence intervals, and local plotting
or dashboard files for learning curves and rollout browsing.
