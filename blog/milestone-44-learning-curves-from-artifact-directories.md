# Milestone 44: Learning Curves From Artifact Directories

Phase 7 now has the missing bridge between online training loops and the
dashboard.

The GRPO and PPO runners already write one directory per cycle: rollouts,
metrics, checkpoints, and optional dev-set evaluation reports. That layout is
exactly what I want for reproducibility, but it is not pleasant to inspect by
hand. This milestone adds a small collector that treats those directories as a
time series.

## What landed

The new module is `learning_curves.py`.

It adds:

- `LearningCurvePoint`;
- `LearningCurveRun`;
- `collect_learning_curve`;
- `collect_learning_curves`.

The collector scans an artifact root for `cycle_*` directories and summarizes
each cycle. For every cycle, it reads the training rollouts, trainer metrics,
checkpoint manifest, and optional `evaluation.json`. The resulting point
records train pass@1, reward mean, degenerate-output rate, response length,
optimizer steps, loss terms, KL terms, and dev-evaluation metrics when those
exist.

The dashboard script now accepts curve inputs too:

```bash
python3 scripts/render_evaluation_dashboard.py \
  --curve grpo=outputs/grpo_online \
  --curve ppo=outputs/ppo_online \
  --evaluation heldout=outputs/evaluation.json \
  --output outputs/dashboard.html
```

The generated static HTML now includes a Learning Curves section with a cycle
table and inline SVG plots for train pass@1, eval pass@1, reward, loss,
entropy, KL, response length, and degenerate-output rate.

## Why this matters

For execution-feedback RL, the final held-out score is only one piece of the
story. I also need to know whether the model is improving smoothly, whether KL
is drifting, whether rewards are increasing while pass rate is flat, and
whether the model is finding degenerate shortcuts.

Before this milestone, answering those questions meant opening many JSONL files
and mentally stitching together the experiment. Now the artifact directory
itself is enough to render a first-pass learning curve.

I kept this dependency-free on purpose. Plotting libraries will be useful later,
but a static HTML artifact that works from a fresh checkout is harder to break
and easier to attach to an experiment bundle.

## What I tested

The unit tests build a miniature online artifact directory with two cycles,
training rollouts, metric rows, checkpoint manifests, and evaluation reports.
They verify the aggregate rollout counts, optimizer steps, pass@1 values, loss
aggregation, approximate KL, serialization, and missing-directory validation.

The dashboard tests now exercise `--curve LABEL=ARTIFACT_DIR` end to end and
check that the generated HTML contains the new Learning Curves section.

## Remaining questions

This is still a first plotting surface. The next useful dashboard improvement
is not another chart, but better browsing: filterable rollout failures, task
families, prompt/source slices, and direct links from a curve anomaly to the
examples that caused it.

There is also one subtle reporting issue. If a trainer did not emit a metric,
the dashboard renders that value as zero. That is simple and deterministic, but
it means the reader must distinguish "measured zero" from "not recorded yet".
The underlying collector keeps optional evaluation fields explicit, so a future
dashboard pass can make missing optimization diagnostics visually distinct.
