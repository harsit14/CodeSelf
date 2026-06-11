#!/usr/bin/env python3
"""Inspect the dependency-free training-core config and backend availability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.config import load_config_file  # noqa: E402
from codeself.training import backend_availability, build_training_core_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON/simple-YAML experiment config.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    raw_config = load_config_file(args.config)
    core_config = build_training_core_config(raw_config)
    availability = backend_availability(core_config.backend)
    payload = {
        "config_path": str(args.config),
        "training_core": core_config.to_dict(),
        "backend_availability": availability.to_dict(),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote training core inspection to {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
