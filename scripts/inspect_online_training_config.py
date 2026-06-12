#!/usr/bin/env python3
"""Inspect and normalize an online GRPO/PPO training config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.config import load_config_file  # noqa: E402
from codeself.training import (  # noqa: E402
    build_online_training_config,
    resolve_online_algorithm,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON/simple-YAML config.")
    parser.add_argument(
        "--algorithm",
        choices=("grpo", "ppo"),
        help="Override the algorithm in the config file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the normalized typed config JSON.",
    )
    args = parser.parse_args()

    raw_config = load_config_file(args.config)
    algorithm = resolve_online_algorithm(raw_config, args.algorithm)
    online_config = build_online_training_config(raw_config, algorithm=algorithm)
    normalized = {
        "algorithm": algorithm,
        "config_path": str(args.config),
        "online_training": online_config.to_dict(),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")

    for key, value in _summary_rows(algorithm, online_config):
        print(f"{key}: {_format_value(value)}")
    if args.output:
        print(f"normalized_output: {args.output}")
    return 0


def _summary_rows(algorithm: str, online_config: Any) -> tuple[tuple[str, object], ...]:
    cycle = online_config.cycle
    rows: list[tuple[str, object]] = [
        ("algorithm", algorithm),
        ("cycles", online_config.cycles),
        ("seed", cycle.seed),
        ("seed_stride", online_config.seed_stride),
        ("rollout_mode", cycle.rollout_mode),
        ("samples_per_task", cycle.rollout.samples_per_task),
        ("group_size", cycle.rollout.group_size),
        ("max_response_tokens", cycle.rollout.max_response_tokens),
        ("self_debug_revision_strategy", cycle.self_debug.revision_strategy),
        ("self_debug_max_revisions", cycle.self_debug.max_revisions),
        ("evaluation_enabled", online_config.evaluation is not None),
        ("sync_old_policy_before_cycle", online_config.sync_old_policy_before_cycle),
    ]
    if algorithm == "ppo":
        rows.append(("sync_old_value_before_cycle", online_config.sync_old_value_before_cycle))
    if online_config.evaluation is not None:
        rows.append(("evaluation_ks", list(online_config.evaluation.ks)))
    return tuple(rows)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
