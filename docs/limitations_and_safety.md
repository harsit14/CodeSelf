# Limitations And Safety

CodeSelf is a bounded research project, not an open-ended autonomous
self-improvement system. The current implementation is a dependency-free
scaffold for coding-task rollouts, rewards, smoke training diagnostics, and
evaluation reports.

## Current Limitations

- The GRPO and PPO trainers are smoke scaffolds and do not update model weights.
- The mock backend is useful for testing plumbing, not for measuring model skill.
- Real training needs model log probabilities, KL accounting, checkpoint writes,
  and GPU-backed generation.
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
