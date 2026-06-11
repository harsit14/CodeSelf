# Docker executor

Milestone 2 defines the clean-container path for final evaluation.

Build:

```bash
docker build -f docker/executor.Dockerfile -t codeself-executor:latest .
```

Runtime contract:

- `--network none`
- `--read-only`
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- non-root user
- memory limit
- memory-swap limit matching memory
- CPU limit
- process limit
- small `/tmp` tmpfs
- bind-mounted task directory
- injected CodeSelf phase harness with candidate and test code

Training rollouts may use the faster subprocess worker path if it preserves the
safety requirements defined in `docs/overhaul_plan.md`. Final evaluation should
prefer clean-container isolation.
