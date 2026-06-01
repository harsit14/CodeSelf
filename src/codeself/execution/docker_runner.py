"""Docker command builder for stricter final-evaluation isolation.

The current milestone does not require Docker to be installed for tests. This
module captures the runtime contract for the final evaluator so the CLI wrapper
can be added once the subprocess path and reward engine are stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeself.datasets import ResourceLimits


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Docker image/runtime settings for final evaluation."""

    image: str = "codeself-executor:latest"
    cpus: float = 1.0
    pids_limit: int = 64
    user: str = "1000:1000"
    workdir: str = "/workspace"


class DockerSandboxRunner:
    """Builds Docker run commands for clean-container evaluation."""

    def __init__(self, config: DockerSandboxConfig | None = None) -> None:
        self.config = config or DockerSandboxConfig()

    def build_command(self, task_dir: str | Path, limits: ResourceLimits) -> list[str]:
        task_path = Path(task_dir).resolve()
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--pids-limit",
            str(self.config.pids_limit),
            "--cpus",
            str(self.config.cpus),
            "--memory",
            f"{limits.memory_mb}m",
            "--user",
            self.config.user,
            "--workdir",
            self.config.workdir,
            "--mount",
            f"type=bind,source={task_path},target={self.config.workdir},readonly",
            self.config.image,
            "python",
            "run_phase.py",
        ]
