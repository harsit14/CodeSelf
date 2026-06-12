# Milestone 46: Model-Generated Self-Debug Revisions

Phase 8 now has the first model-driven revision path.

The previous milestone made self-debug traces usable as ordinary rollout
records. This milestone changes what can happen inside the trace. Instead of
only applying the tiny rule-based repair helper, the agent loop can now ask the
generator for a revised solution after public-test feedback.

## What landed

I added a versioned revision prompt template:

- `RevisionPromptTemplate`;
- `SELF_DEBUG_REVISION_TEMPLATE`;
- `get_revision_prompt_template`.

The default prompt is intentionally explicit. It includes the task, required
entry point, starter code, public tests, current code, and the observed
public-test feedback. It does not include hidden tests. The instruction is also
deliberately boring: return a complete corrected solution, not a patch.

`SelfDebugAgentLoop` now supports a model revision branch. When public tests
fail and `revision_strategy` is `model`, the loop creates a second
`GenerationRequest` using the revision prompt. The resulting completion is
parsed into the next candidate program, then public tests run again as before.

## Trace metadata

The revision step now records enough context to audit the behavior later:

- revision source;
- revision prompt template;
- revision prompt text;
- raw revision completion;
- backend;
- model name;
- generator metadata.

That makes self-debug training more inspectable. If a revised solution suddenly
passes, I can look at the exact feedback prompt that produced it.

## CLI and configs

Both rollout and trace-only scripts now expose the strategy:

```bash
python3 scripts/run_rollouts.py \
  --rollout-mode self_debug \
  --revision-strategy model \
  --tasks configs/datasets/tasks.example.jsonl \
  --output outputs/rollouts/self_debug_model_revision.jsonl
```

The supported strategies are:

- `rule_based`;
- `model`;
- `none`.

I also added `rollout_self_debug_model_revision.example.yaml` so the model
revision path has a concrete ablation config next to the single-shot and
rule-based self-debug configs.

## What I tested

The tests use a small two-stage generator: it emits a broken first solution and
only emits the corrected solution when it sees the structured revision prompt.
That verifies the loop is really exercising the model-revision branch, not the
old rule-based helper.

The tests also check that hidden tests do not leak into the revision prompt,
that trace metadata records the revision prompt and source, and that the final
result still becomes a normal rollout record with `revision_strategy=model`.

## What is next

This is enough to run self-debug ablations, but not enough to make the online
trainers feel native. The next useful slice is to let GRPO and PPO collect
self-debug rollouts directly inside each online cycle, rather than asking a
researcher to pre-generate rollout JSONL outside the loop.
