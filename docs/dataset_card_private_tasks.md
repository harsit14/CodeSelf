# Dataset Card Template For Private Coding Tasks

This card should be completed for any private task set used in CodeSelf. The
example repository only includes a toy task file, so the fields below are a
publication template rather than a claim about an existing private dataset.

## Dataset Summary

- Dataset name: TBD.
- Version: TBD.
- Maintainer: TBD.
- Task language: Python.
- Task format: CodeSelf canonical JSONL.
- Number of tasks: TBD.
- Splits: train, dev, public test, private test, stress.
- License: TBD.

## Sources

Describe where tasks came from, who authored them, and whether they were
adapted from public benchmarks. If tasks are based on public data, include the
original license and a link to the source.

## Splitting Policy

Splits must be assigned before model training starts. Private test tasks should
remain untouched until the final locked evaluation. Any split regeneration must
produce a new manifest and invalidate previous final results.

## Hidden Tests

Hidden tests are used for scoring, not prompting. They must not appear in:

- task prompts
- starter code
- public examples
- rollout records shown to the model
- blog posts or public demos before final release

## Contamination Controls

- [ ] Search for exact prompt overlap with common public benchmarks.
- [ ] Search for exact solution overlap when reference solutions exist.
- [ ] Record dataset creation dates and source URLs.
- [ ] Record model training cutoff assumptions when available.
- [ ] Keep public-test tuning separate from private-test evaluation.

## Known Biases

Document task difficulty, topic coverage, missing domains, and whether the
tasks overrepresent short algorithmic functions relative to real software
engineering.

## Publication Decision

State whether the dataset can be published. If not, publish aggregate
statistics, split hashes, license rationale, and the evaluation protocol so the
results remain inspectable even when raw tasks stay private.
