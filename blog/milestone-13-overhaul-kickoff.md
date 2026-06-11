# Milestone 13: Starting The Real CodeSelf Overhaul

Today I started the part of CodeSelf that feels like switching from a clean
systems sketch to an actual research instrument.

The project already had a surprisingly useful smoke pipeline. It could load
coding tasks, sample mock solutions, execute them, score them, write rollout
records, run GRPO/PPO-style diagnostics, and produce evaluation reports. But the
important caveat was always sitting in the README: the trainers did not update
model weights. They proved the shape of the experiment, not the learning.

This overhaul is about closing that gap without losing the thing that made the
scaffold valuable: fast, dependency-free tests.

## The machine constraint is real

The local machine for this work is a MacBook M5 Pro with 48GB unified memory.
That is a very good development box, but it changes how I think about the
training stack. I do not want the whole project to assume a CUDA server just to
run a unit test or inspect a rollout. At the same time, the main research run
should be portable to a standard single-GPU setup where bf16 LoRA training on a
small coder model is realistic.

So the plan is split in two:

- keep local smoke and debug runs light enough for the Mac;
- design the real GRPO/PPO path around a small open coder model, probably in
  the Qwen2.5-Coder 0.5B or 1.5B class, with PEFT/LoRA and a config switch for
  full fine-tuning.

This is also why I am treating backends as an architectural decision, not an
implementation detail. A from-scratch trainer is fine for the first real pass,
but it should be possible to swap in TRL or verl later without rewriting the
data, sandbox, reward, and evaluation layers.

## The first engineering move

Before touching the sandbox or RL loss, I made the smoke boundary explicit.
The original GRPO and PPO diagnostic trainers now belong under a
`training/smoke/` namespace. The old import paths stay alive through thin
compatibility wrappers, so existing scripts and tests do not need to care about
the move.

This is a small change, but it matters. I want future code readers to be able to
answer a basic question quickly: "Does this class train a model, or does it only
validate the pipeline?" Before this phase, the answer lived in docstrings. Now
it also lives in the package structure.

## Why I am not starting with the loss function

It is tempting to jump straight into GRPO because that is the flashy part. But
execution-feedback RL for code has a trap: if the executor is soft, the reward
is soft. A model can learn against quirks in the harness, leak expected outputs,
or get credit from failure modes that were accidentally collapsed into the same
label.

So after this documentation and smoke-preservation phase, the next target is the
sandbox. I want per-test outcomes, stronger process cleanup, adversarial tests,
and a final-evaluation path that is container-ready. Only then does it make
sense to feed rewards into a real policy-gradient loop.

## The research diary plan

I am going to keep writing these milestone notes as the overhaul proceeds. The
tone is intentionally practical: enough background for another CS grad student
to understand the design choices, but close enough to the code that the notes
remain falsifiable.

The rough sequence is:

1. preserve smoke behavior and document the plan;
2. harden the sandbox;
3. generalize datasets and rewards;
4. build the common RL rollout/training core;
5. implement real GRPO;
6. add PPO as a matched baseline;
7. upgrade evaluation and plotting;
8. make self-debugging a first-class training mode;
9. rewrite the public docs around actual results.

The north star is modest but serious: not "recursive self-improvement," not a
demo curve from a fragile benchmark, but a reproducible experiment showing
whether execution feedback can improve a small coding model on a declared Python
task distribution.
