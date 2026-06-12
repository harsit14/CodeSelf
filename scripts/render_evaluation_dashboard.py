#!/usr/bin/env python3
"""Render a static HTML dashboard from evaluation/comparison JSON reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeself.evaluation import (  # noqa: E402
    collect_learning_curve,
    read_dashboard_json,
    write_evaluation_dashboard,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path.")
    parser.add_argument("--title", default="CodeSelf Evaluation Dashboard")
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Evaluation JSON report. May be passed multiple times.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Comparison JSON report. May be passed multiple times.",
    )
    parser.add_argument(
        "--curve",
        action="append",
        default=[],
        metavar="LABEL=ARTIFACT_DIR",
        help="Online training artifact directory with cycle_* children.",
    )
    args = parser.parse_args()

    evaluations = _read_labeled_reports(args.evaluation)
    comparisons = _read_labeled_reports(args.comparison)
    curves = _read_labeled_curves(args.curve)
    if not evaluations and not comparisons and not curves:
        parser.error("provide at least one --evaluation, --comparison, or --curve")

    write_evaluation_dashboard(
        args.output,
        title=args.title,
        evaluations=evaluations,
        comparisons=comparisons,
        curves=curves,
    )
    print(f"wrote evaluation dashboard to {args.output}")
    print(f"evaluation_reports: {len(evaluations)}")
    print(f"comparison_reports: {len(comparisons)}")
    print(f"curve_reports: {len(curves)}")
    return 0


def _read_labeled_reports(values: list[str]) -> dict[str, dict[str, object]]:
    reports: dict[str, dict[str, object]] = {}
    for value in values:
        label, path = _parse_labeled_path(value)
        if label in reports:
            raise ValueError(f"duplicate report label: {label}")
        reports[label] = read_dashboard_json(path)
    return reports


def _read_labeled_curves(values: list[str]) -> dict[str, dict[str, object]]:
    curves: dict[str, dict[str, object]] = {}
    for value in values:
        label, path = _parse_labeled_path(value)
        if label in curves:
            raise ValueError(f"duplicate curve label: {label}")
        curves[label] = collect_learning_curve(label, path).to_dict()
    return curves


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path_text = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("report label must not be empty")
        return label, Path(path_text)
    path = Path(value)
    return path.stem, path


if __name__ == "__main__":
    raise SystemExit(main())
