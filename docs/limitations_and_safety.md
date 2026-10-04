# Limitations And Safety

CodeSelf is a bounded research project, not an open-ended autonomous
self-improvement system. It provides a dependency-free pipeline for
coding-task rollouts, rewards, and evaluation reports, plus optional real
GRPO/PPO LoRA training through the `training` extra.

## Current Limitations

- Real GRPO/PPO training has been verified only at debug scale (12 tasks,
  a 0.6B base model, one device, one seed). Reported gains are pipeline
  validation, not benchmark results.
- The mock backend is useful for testing plumbing, not for measuring model skill.
- PPO did not improve in the debug setting; whether that holds with a warmed-up
  critic and larger batches is untested.
- Benchmark contamination can make pass rates look better than genuine
  generalization.
- Small evaluation sets can be underpowered.
- Public tests can be overfit if they are used repeatedly for revisions.
- Quality and efficiency are diagnostics in the MVP, not optimized objectives.
- Agentic strategy claims require trace evidence from tool use, not just final
  answer improvements.

## Safety Principles

- Treat generated code as hostile until it passes sandbox checks.
- Keep hidden tests out of prompts and public rollout traces.
- Run final evaluation in stricter isolation than training smoke tests.
- Avoid publishing private datasets or hidden tests without license clearance.
- Avoid claiming general recursive self-improvement.
- Report failures, regressions, parser errors, and sandbox errors alongside
  successful examples.

## Release Risks

An adapter trained on execution rewards could learn brittle benchmark-specific
patterns, produce unsafe code, or exploit reward loopholes. A public release
should include the reward version, limitations, known failure modes, and
instructions to execute generated code only inside a restricted environment.
