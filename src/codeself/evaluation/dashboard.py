"""Static HTML dashboards for evaluation and comparison reports."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


JsonMapping = Mapping[str, Any]


def read_dashboard_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON report file for dashboard rendering."""

    input_path = Path(path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dashboard JSON: {input_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"dashboard JSON must contain an object: {input_path}")
    return payload


def render_evaluation_dashboard(
    *,
    title: str,
    evaluations: Mapping[str, JsonMapping],
    comparisons: Mapping[str, JsonMapping] | None = None,
) -> str:
    """Render a static dashboard from evaluation and comparison JSON payloads."""

    if not evaluations and not comparisons:
        raise ValueError("dashboard requires at least one evaluation or comparison")
    comparison_payloads = comparisons or {}
    safe_title = _escape(title)
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{safe_title}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{safe_title}</h1>",
        '<p class="lede">Evaluation, statistics, and rollout browsing summary.</p>',
    ]
    if evaluations:
        parts.extend(_render_evaluation_section(evaluations))
    if comparison_payloads:
        parts.extend(_render_comparison_section(comparison_payloads))
    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts) + "\n"


def write_evaluation_dashboard(
    output_path: str | Path,
    *,
    title: str,
    evaluations: Mapping[str, JsonMapping],
    comparisons: Mapping[str, JsonMapping] | None = None,
) -> None:
    """Write a static HTML dashboard file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_evaluation_dashboard(
            title=title,
            evaluations=evaluations,
            comparisons=comparisons,
        ),
        encoding="utf-8",
    )


def _render_evaluation_section(evaluations: Mapping[str, JsonMapping]) -> list[str]:
    rows = []
    for label, report in evaluations.items():
        rows.append(
            [
                label,
                _int(report, "rollout_count"),
                _int(report, "task_count"),
                _float(report, "pass_at.pass@1"),
                _float(report, "execution_pass_rate"),
                _float(report, "parse_failure_rate"),
                _float(report, "degenerate_output_rate"),
                _float(report, "reward_mean"),
                _float(report, "response_token_mean"),
            ]
        )
    output = [
        '<section class="band">',
        "<h2>Evaluation Runs</h2>",
        _table(
            (
                "Run",
                "Rollouts",
                "Tasks",
                "pass@1",
                "Pass Rate",
                "Parse Fail",
                "Degenerate",
                "Mean Reward",
                "Mean Tokens",
            ),
            rows,
            numeric_columns={1, 2, 3, 4, 5, 6, 7, 8},
        ),
        "<h3>pass@1 By Run</h3>",
        _bar_chart(
            [
                (label, _float(report, "pass_at.pass@1"))
                for label, report in evaluations.items()
            ],
            maximum=1.0,
        ),
        "<h3>Reward And Length</h3>",
        _table(
            (
                "Run",
                "Reward Mean",
                "Reward Std",
                "Reward Min",
                "Reward Max",
                "Token Mean",
                "Token Median",
                "Token Max",
            ),
            [
                [
                    label,
                    _float(report, "reward_mean"),
                    _float(report, "reward_std"),
                    _float(report, "reward_min"),
                    _float(report, "reward_max"),
                    _float(report, "response_token_mean"),
                    _float(report, "response_token_median"),
                    _int(report, "response_token_max"),
                ]
                for label, report in evaluations.items()
            ],
            numeric_columns={1, 2, 3, 4, 5, 6, 7},
        ),
    ]
    output.extend(_render_group_tables(evaluations))
    output.extend(_render_task_tables(evaluations))
    output.append("</section>")
    return output


def _render_group_tables(evaluations: Mapping[str, JsonMapping]) -> list[str]:
    rows = []
    for label, report in evaluations.items():
        for group in _list(report, "group_summaries"):
            rows.append(
                [
                    label,
                    str(group.get("group_field", "")),
                    str(group.get("group_value", "")),
                    int(group.get("rollout_count", 0)),
                    int(group.get("task_count", 0)),
                    _float(group, "pass_at.pass@1"),
                    float(group.get("execution_pass_rate", 0.0)),
                    float(group.get("degenerate_output_rate", 0.0)),
                    float(group.get("reward_mean", 0.0)),
                    float(group.get("response_token_mean", 0.0)),
                ]
            )
    if not rows:
        return []
    return [
        "<h3>Group Metrics</h3>",
        _table(
            (
                "Run",
                "Field",
                "Value",
                "Rollouts",
                "Tasks",
                "pass@1",
                "Pass Rate",
                "Degenerate",
                "Reward",
                "Tokens",
            ),
            rows,
            numeric_columns={3, 4, 5, 6, 7, 8, 9},
        ),
    ]


def _render_task_tables(evaluations: Mapping[str, JsonMapping]) -> list[str]:
    rows = []
    for label, report in evaluations.items():
        for task in _list(report, "task_summaries"):
            rows.append(
                [
                    label,
                    str(task.get("task_id", "")),
                    int(task.get("total_samples", 0)),
                    int(task.get("correct_samples", 0)),
                    int(task.get("parse_failures", 0)),
                    _float(task, "pass_at.pass@1"),
                    float(task.get("degenerate_output_rate", 0.0)),
                    float(task.get("mean_reward", 0.0)),
                    float(task.get("response_token_mean", 0.0)),
                ]
            )
    if not rows:
        return []
    return [
        "<h3>Task Browser</h3>",
        _table(
            (
                "Run",
                "Task",
                "Samples",
                "Correct",
                "Parse Fail",
                "pass@1",
                "Degenerate",
                "Reward",
                "Tokens",
            ),
            rows,
            numeric_columns={2, 3, 4, 5, 6, 7, 8},
        ),
    ]


def _render_comparison_section(comparisons: Mapping[str, JsonMapping]) -> list[str]:
    summary_rows = []
    flip_rows = []
    for label, report in comparisons.items():
        effect = (
            report.get("effect_size", {})
            if isinstance(report.get("effect_size"), dict)
            else {}
        )
        permutation = (
            report.get("permutation", {}) if isinstance(report.get("permutation"), dict) else {}
        )
        bootstrap = (
            report.get("bootstrap_delta", {})
            if isinstance(report.get("bootstrap_delta"), dict)
            else {}
        )
        mcnemar = report.get("mcnemar", {}) if isinstance(report.get("mcnemar"), dict) else {}
        summary_rows.append(
            [
                label,
                int(report.get("task_count", 0)),
                float(report.get("base_pass_rate", 0.0)),
                float(report.get("candidate_pass_rate", 0.0)),
                float(report.get("delta_pp", 0.0)),
                float(mcnemar.get("p_value", 0.0)),
                float(permutation.get("p_value", 0.0)),
                float(bootstrap.get("lower", 0.0)) * 100,
                float(bootstrap.get("upper", 0.0)) * 100,
                _optional_number(effect.get("relative_error_reduction")),
            ]
        )
        for outcome in _list(report, "outcomes"):
            delta = int(outcome.get("delta", 0))
            if delta == 0:
                continue
            flip_rows.append(
                [
                    label,
                    str(outcome.get("task_id", "")),
                    int(bool(outcome.get("base_passed", False))),
                    int(bool(outcome.get("candidate_passed", False))),
                    float(outcome.get("base_reward", 0.0)),
                    float(outcome.get("candidate_reward", 0.0)),
                    delta,
                ]
            )
    output = [
        '<section class="band">',
        "<h2>Paired Comparisons</h2>",
        _table(
            (
                "Comparison",
                "Tasks",
                "Base pass@1",
                "Candidate pass@1",
                "Delta pp",
                "McNemar p",
                "Permutation p",
                "CI Low pp",
                "CI High pp",
                "Error Reduction",
            ),
            summary_rows,
            numeric_columns={1, 2, 3, 4, 5, 6, 7, 8, 9},
        ),
    ]
    if summary_rows:
        output.extend(
            [
                "<h3>Delta By Comparison</h3>",
                _bar_chart(
                    [(row[0], float(row[4])) for row in summary_rows],
                    maximum=max(1.0, max(abs(float(row[4])) for row in summary_rows)),
                ),
            ]
        )
    if flip_rows:
        output.extend(
            [
                "<h3>Task Flips</h3>",
                _table(
                    (
                        "Comparison",
                        "Task",
                        "Base",
                        "Candidate",
                        "Base Reward",
                        "Candidate Reward",
                        "Delta",
                    ),
                    flip_rows,
                    numeric_columns={2, 3, 4, 5, 6},
                ),
            ]
        )
    output.append("</section>")
    return output


def _table(
    headers: tuple[str, ...],
    rows: list[list[object]],
    *,
    numeric_columns: set[int],
) -> str:
    parts = ['<div class="table-wrap"><table>']
    parts.append(
        "<thead><tr>"
        + "".join(f"<th>{_escape(header)}</th>" for header in headers)
        + "</tr></thead>"
    )
    parts.append("<tbody>")
    if not rows:
        parts.append(f'<tr><td colspan="{len(headers)}">No rows.</td></tr>')
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            class_name = ' class="num"' if index in numeric_columns else ""
            cells.append(f"<td{class_name}>{_format_cell(value)}</td>")
        parts.append("<tr>" + "".join(cells) + "</tr>")
    parts.extend(["</tbody>", "</table></div>"])
    return "\n".join(parts)


def _bar_chart(values: list[tuple[str, float]], *, maximum: float) -> str:
    width = 720
    row_height = 30
    left = 180
    right = 80
    height = max(60, 24 + row_height * len(values))
    chart_width = width - left - right
    max_value = maximum if maximum > 0 else 1.0
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">',
        '<line class="axis" x1="{0}" y1="12" x2="{0}" y2="{1}"></line>'.format(
            left,
            height - 12,
        ),
    ]
    for index, (label, value) in enumerate(values):
        y = 18 + index * row_height
        bar_width = abs(value) / max_value * chart_width
        x = left if value >= 0 else left - bar_width
        parts.append(
            f'<text class="chart-label" x="8" y="{y + 14}">{_escape(label)}</text>'
        )
        parts.append(
            f'<rect class="bar" x="{x:.2f}" y="{y}" '
            f'width="{bar_width:.2f}" height="18"></rect>'
        )
        parts.append(
            f'<text class="chart-value" x="{left + chart_width + 8}" '
            f'y="{y + 14}">{value:.4g}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _int(payload: JsonMapping, path: str) -> int:
    value = _lookup(payload, path)
    return int(value) if isinstance(value, (int, float)) else 0


def _float(payload: JsonMapping, path: str) -> float:
    value = _lookup(payload, path)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _list(payload: JsonMapping, path: str) -> list[dict[str, Any]]:
    value = _lookup(payload, path)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _lookup(payload: JsonMapping, path: str) -> object | None:
    value: object = payload
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _optional_number(value: object) -> float | str:
    return float(value) if isinstance(value, (int, float)) else "n/a"


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return _escape(str(value))


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


_CSS = """
:root {
  color-scheme: light;
  --bg: #f7f8fa;
  --ink: #16202a;
  --muted: #5f6b76;
  --line: #cfd6dd;
  --accent: #1769aa;
  --accent-soft: #d9ecf8;
  --band: #ffffff;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
main {
  max-width: 1180px;
  margin: 0 auto;
  padding: 28px 18px 42px;
}
h1 {
  margin: 0;
  font-size: 28px;
  letter-spacing: 0;
}
h2 {
  margin: 0 0 14px;
  font-size: 20px;
  letter-spacing: 0;
}
h3 {
  margin: 22px 0 10px;
  font-size: 15px;
  letter-spacing: 0;
}
.lede {
  margin: 6px 0 20px;
  color: var(--muted);
}
.band {
  background: var(--band);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin: 16px 0;
  padding: 16px;
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 760px;
}
th,
td {
  padding: 8px 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}
th {
  background: #eef2f5;
  color: #27313b;
  font-weight: 650;
}
tr:last-child td {
  border-bottom: 0;
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.chart {
  display: block;
  width: 100%;
  max-width: 760px;
  min-height: 80px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: #fbfcfd;
}
.bar {
  fill: var(--accent);
}
.axis {
  stroke: var(--line);
}
.chart-label,
.chart-value {
  fill: var(--ink);
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}
@media (max-width: 720px) {
  main {
    padding: 18px 10px 32px;
  }
  .band {
    padding: 12px;
  }
  h1 {
    font-size: 24px;
  }
}
"""
