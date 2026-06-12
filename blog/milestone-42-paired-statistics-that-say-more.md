# Milestone 42: Paired Statistics That Say More

Phase 7 now has stronger paired comparison reports.

The previous slice made evaluation summaries more dashboard-ready. This one
focuses on final comparisons: when GRPO, PPO, or a baseline are evaluated on the
same tasks, the report should say more than "candidate pass@1 is higher."

## What landed

The main changes are in `statistical_tests.py`.

I added:

- `PairedPermutationResult`;
- `paired_permutation_test`;
- `EffectSizeSummary`;
- `effect_size_summary`.

`ComparisonSummary` now carries four views of the same paired comparison:

- exact McNemar/binomial p-value over discordant pass/fail pairs;
- paired sign-flip permutation p-value over task deltas;
- bootstrap confidence interval over pass-rate delta;
- effect-size fields that are easier to read in a paper table.

## Exact when small, sampled when large

The permutation test is dependency-free. For small discordant task sets, it
enumerates every sign flip exactly. For larger comparisons, it uses a seeded
Monte Carlo estimate. That keeps the result reproducible without pulling in
SciPy.

## Effect sizes

The report now includes:

- pass-rate delta;
- percentage-point delta;
- relative pass-rate lift;
- base and candidate error rates;
- relative error reduction;
- reward delta;
- standardized paired reward delta.

The p-value answers "could this happen under a symmetric null?" The effect size
answers "how large is the change?" We need both.

## CLI changes

`scripts/compare_rollouts.py` now accepts `--permutation-samples` and prints the
permutation p-value plus the effect-size highlights. JSON and Markdown reports
carry the same fields, so later dashboard code can consume them directly.

The next Phase 7 work should turn these richer report payloads into local plots
and browsing views for learning curves, grouped pass rates, reward trends, KL,
entropy, length, and degenerate-output rates.
