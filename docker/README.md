# Docker executor

Milestone 2 defines the clean-container path for final evaluation.

Build:

```bash
docker build -f docker/executor.Dockerfile -t codeself-executor:latest .
```

Runtime contract:

- `--network none`
- `--read-only`
- non-root user
- memory limit
- CPU limit
- process limit
- bind-mounted task directory

Training rollouts may use the faster subprocess worker path if it preserves the safety requirements defined in `plan.md`. Final evaluation should prefer clean-container isolation.
