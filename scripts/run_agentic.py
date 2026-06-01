#!/usr/bin/env python3
"""Run the public-test self-debug agent loop on canonical tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.agent import (  # noqa: E402
    AgentLoopConfig,
    SelfDebugAgentLoop,
    analyze_traces,
    get_prompt_template,
    make_generator,
    run_agentic_tasks,
    write_agent_traces_jsonl,
    write_strategy_report,
)
from codeself.datasets import Split, TaskRegistry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Canonical task JSONL file.")
    parser.add_argument("--trace-output", type=Path, required=True, help="Output agent traces JSONL.")
    parser.add_argument("--report-output", type=Path, required=True, help="Output strategy report .md/.json.")
    parser.add_argument("--backend", choices=("mock", "static", "transformers"), default="mock")
    parser.add_argument("--model")
    parser.add_argument("--static-completion-file", type=Path)
    parser.add_argument("--prompt-template", default="direct_solution_v1")
    parser.add_argument("--split", choices=[split.value for split in Split])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max-revisions", type=int, default=1)
    parser.add_argument("--disable-rule-repair", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()

    registry = TaskRegistry.from_jsonl(args.tasks)
    tasks = registry.by_split(args.split) if args.split else list(registry)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no tasks selected")

    static_completion = (
        args.static_completion_file.read_text(encoding="utf-8")
        if args.static_completion_file
        else None
    )
    generator = make_generator(
        args.backend,
        model_name_or_path=args.model,
        static_completion=static_completion,
    )
    loop = SelfDebugAgentLoop(
        generator=generator,
        prompt_template=get_prompt_template(args.prompt_template),
        config=AgentLoopConfig(
            max_revisions=args.max_revisions,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            use_rule_based_repair=not args.disable_rule_repair,
        ),
    )
    traces = run_agentic_tasks(tasks, loop=loop, sample_index=args.sample_index)
    analysis = analyze_traces(traces)
    write_agent_traces_jsonl(traces, args.trace_output)
    write_strategy_report(analysis, args.report_output)

    print(f"wrote {len(traces)} agent traces to {args.trace_output}")
    print(f"wrote strategy report to {args.report_output}")
    print(f"final_pass_rate: {analysis.final_pass_rate:.4f}")
    print(f"mean_reward: {analysis.mean_reward:.4f}")
    print(f"revision_rate: {analysis.revision_rate:.4f}")
    print(f"mean_tool_calls: {analysis.mean_tool_calls:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
