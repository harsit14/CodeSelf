#!/usr/bin/env python3
"""Render one agent trace as Markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import read_agent_traces_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--task-id", help="Task ID to render. Defaults to first trace.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    traces = read_agent_traces_jsonl(args.traces)
    if not traces:
        raise SystemExit("no traces found")
    trace = next((item for item in traces if item.task_id == args.task_id), traces[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_trace_markdown(trace), encoding="utf-8")
    print(f"wrote trace view to {args.output}")
    return 0


def _trace_markdown(trace) -> str:
    lines = [
        "# Agent Trace",
        "",
        f"- Task: {trace.task_id}",
        f"- Final passed: {trace.final_passed}",
        f"- Revisions: {trace.revision_count}",
        f"- Tool calls: {trace.tool_call_count}",
        f"- Reward: {float(trace.final_reward.get('reward', 0.0)):.4f}",
        "",
        "## Steps",
        "",
    ]
    for index, step in enumerate(trace.steps, start=1):
        lines.append(f"### {index}. {step.kind}")
        if step.observation:
            lines.append("")
            lines.append(step.observation)
        if step.code:
            lines.append("")
            lines.append("```python")
            lines.append(step.code.strip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
