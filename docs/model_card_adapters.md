# Adapter Model Card Template

CodeSelf does not currently publish trained adapter weights. This template is
the card to fill in before releasing any LoRA or full fine-tuned checkpoint.

## Model Summary

- Adapter name: TBD.
- Base model: TBD.
- Base model revision: TBD.
- Tokenizer revision: TBD.
- Training algorithm: GRPO primary, PPO baseline if run.
- Adapter type: TBD.
- Parameter count: TBD.
- Release date: TBD.
- License: TBD.

## Intended Use

The adapter is intended for research on execution-feedback reinforcement
learning for Python coding tasks. It is not intended for unsupervised deployment,
security-sensitive code generation, or claims of general recursive
self-improvement.

## Training Data

Training tasks should be described using the dataset card in
`docs/dataset_card_private_tasks.md`. The release should list public benchmark
sources, private task sources if publishable, split hashes, and contamination
controls.

## Reward And Optimization

The MVP reward is `reward_v0_correctness`. It gives primary weight to syntax,
importability, public tests, hidden tests, robustness tests, and safety
penalties. Quality and efficiency metrics are logged as diagnostics in the MVP
and should not be described as optimized objectives unless the reward version
changes.

## Evaluation

Report the following before releasing weights:

- Held-out pass@1 and pass@k.
- Paired base-versus-adapter comparison.
- Bootstrap confidence interval for pass@1 delta.
- Exact McNemar/binomial p-value over discordant task pairs.
- Parser failure rate.
- Sandbox failure rate.
- Mean latency or runtime per task.
- General-coding probe results, if run.
- Regression checks against the base model.

## Safety And Limitations

Generated code is untrusted and must be sandboxed before execution. The adapter
may memorize benchmark patterns, exploit reward loopholes, fail on tasks outside
the training distribution, or regress on general coding ability. Any public
release should include examples of failures, not only successful traces.

## Release Checklist

- [ ] Base model license permits adapter release.
- [ ] Dataset licenses permit this use.
- [ ] No private hidden tests are included in public files.
- [ ] Adapter checkpoint checksum is published.
- [ ] Training config and evaluation config are published.
- [ ] Reproducibility manifest is published.
- [ ] Limitations and safety section is linked from the README.
