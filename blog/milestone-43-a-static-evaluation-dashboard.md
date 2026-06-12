# Milestone 43: A Static Evaluation Dashboard

Phase 7 now has a local dashboard artifact.

The previous two milestones made the evaluation and comparison JSON payloads
more useful. This milestone turns those payloads into something a researcher can
open in a browser without running a web server.

## What landed

The new module is `dashboard.py`.

It adds:

- `read_dashboard_json`;
- `render_evaluation_dashboard`;
- `write_evaluation_dashboard`.

There is also a new script:

```bash
python3 scripts/render_evaluation_dashboard.py \
  --evaluation heldout=outputs/evaluation.json \
  --comparison grpo_vs_ppo=outputs/comparison.json \
  --output outputs/dashboard.html
```

The output is a static HTML file with inline CSS and SVG. No JavaScript build
step, no Python web server, no plotting dependency.

## What the dashboard shows

The first version includes:

- evaluation summary tables;
- pass@1 bar charts;
- reward and response-length summaries;
- grouped metrics from `group_by` reports;
- per-task browsing tables;
- paired-comparison tables;
- bootstrap confidence interval columns;
- permutation p-values;
- task-flip tables.

This is intentionally compact and utilitarian. It is meant to help inspect
training and evaluation artifacts, not to be a product landing page.

## Why static HTML

Static HTML is boring in the best possible way. It can be checked into an
artifact directory, opened locally, copied into an experiment bundle, or shared
with a reviewer. It also keeps Phase 7 dependency-free for now.

The next dashboard slice should aggregate multi-cycle GRPO/PPO artifact
directories into learning curves: reward, pass rate, KL, entropy, response
length, and degenerate-output fraction over time.
