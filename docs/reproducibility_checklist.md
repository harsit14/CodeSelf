# Reproducibility Checklist

This checklist is the public handoff contract for CodeSelf. A reviewer should be
able to rerun the smoke pipeline now, and a future reviewer should be able to
rerun real GRPO/PPO experiments after model-training dependencies are installed.

## Minimum Local Smoke Reproduction

- [ ] Use Python 3.11 or newer.
- [ ] Start from a clean checkout of the repository.
- [ ] Run `python3 scripts/validate_task_schema.py configs/datasets/tasks.example.jsonl`.
- [ ] Run `python3 -m unittest discover -s tests`.
- [ ] Run a mock rollout with `scripts/run_rollouts.py`.
- [ ] Evaluate that rollout with `scripts/evaluate_rollouts.py`.
- [ ] Run `scripts/train_grpo_smoke.py`.
- [ ] Run `scripts/train_ppo_smoke.py`.
- [ ] Compare smoke metrics with `scripts/compare_training_smoke.py`.
- [ ] Create a manifest with `scripts/make_reproducibility_manifest.py`.

## Required Experiment Metadata

- [ ] Git commit hash.
- [ ] Git dirty/untracked status.
- [ ] Python version.
- [ ] Package version.
- [ ] Model name and exact revision.
- [ ] Tokenizer name and exact revision.
- [ ] Adapter checkpoint checksum.
- [ ] Dataset source and license notes.
- [ ] Split manifest checksum.
- [ ] Prompt template name and hash.
- [ ] Reward version and config hash.
- [ ] Sandbox image or executor hash.
- [ ] Training config.
- [ ] Evaluation config.
- [ ] Random seeds.
- [ ] Hardware type, GPU count, CPU count, memory, and wall-clock time.
- [ ] Raw rollout JSONL files.
- [ ] Final paired statistical report.

## Pretraining And Fine-Tuning Controls

- [ ] Hidden tests are not present in prompts, starter code, or public examples.
- [ ] Held-out tasks are declared before training starts.
- [ ] Private test results are not used for checkpoint selection.
- [ ] The selected checkpoint is chosen from train/dev metrics only.
- [ ] The base model and candidate model are evaluated on the same task IDs.
- [ ] Pass@1 comparisons use paired samples where possible.
- [ ] Multiple comparison corrections are documented when many variants are tested.

## Archive Command

```bash
python3 scripts/make_reproducibility_manifest.py \
  --output outputs/reports/reproducibility_manifest.json \
  --markdown-output outputs/reports/reproducibility_manifest.md \
  --archive outputs/reports/codeself_reproducibility.tar.gz
```

The archive contains public project files plus a manifest of SHA-256 checksums.
Private datasets, hidden tests, model checkpoints, and raw rollouts should be
stored in a separate controlled archive when they cannot be published.
