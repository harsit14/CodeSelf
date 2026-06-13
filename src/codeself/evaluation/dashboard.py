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
    curves: Mapping[str, JsonMapping] | None = None,
    rollouts: Mapping[str, list[JsonMapping]] | None = None,
) -> str:
    """Render a static dashboard from evaluation and comparison JSON payloads."""

    rollout_payloads = rollouts or {}
    if not evaluations and not comparisons and not curves and not rollout_payloads:
        raise ValueError("dashboard requires at least one report, curve, or rollout set")
    comparison_payloads = comparisons or {}
    curve_payloads = curves or {}
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
    if curve_payloads:
        parts.extend(_render_curve_section(curve_payloads))
    if rollout_payloads:
        parts.extend(_render_rollout_browser(rollout_payloads))
    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts) + "\n"


def write_evaluation_dashboard(
    output_path: str | Path,
    *,
    title: str,
    evaluations: Mapping[str, JsonMapping],
    comparisons: Mapping[str, JsonMapping] | None = None,
    curves: Mapping[str, JsonMapping] | None = None,
    rollouts: Mapping[str, list[JsonMapping]] | None = None,
) -> None:
    """Write a static HTML dashboard file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_evaluation_dashboard(
            title=title,
            evaluations=evaluations,
            comparisons=comparisons,
            curves=curves,
            rollouts=rollouts,
        ),
        encoding="utf-8",
    )


def _render_rollout_browser(rollouts: Mapping[str, list[JsonMapping]]) -> list[str]:
    """Render a collapsible browser of individual sampled programs.

    Each rollout shows its task, sample index, reward, pass/fail, any
    reward-hacking flag, the generated code, and (when present) a compact
    summary of per-test outcomes and the self-debug revision count.
    """

    output = ['<section class="band">', "<h2>Rollout Browser</h2>"]
    for label, records in rollouts.items():
        output.append(f"<h3>{_escape(label)} ({len(records)} rollouts)</h3>")
        for record in records:
            task_id = _escape(str(record.get("task_id", "?")))
            sample_index = _escape(str(record.get("sample_index", "?")))
            reward = _format_cell(_lookup(record, "reward.reward"))
            execution = record.get("execution") if isinstance(record, dict) else {}
            passed = bool(execution.get("passed")) if isinstance(execution, dict) else False
            metadata = record.get("metadata") if isinstance(record, dict) else {}
            flagged = bool(metadata.get("reward_hacking_flagged")) if isinstance(metadata, dict) else False
            difficulty = _escape(str(metadata.get("difficulty", "?"))) if isinstance(metadata, dict) else "?"
            revisions = (
                _lookup(record, "reward.metrics.self_debug_revision_count")
                if isinstance(record, dict)
                else None
            )
            status = "pass" if passed else "fail"
            badges = [f'<span class="badge {status}">{status}</span>']
            if flagged:
                badges.append('<span class="badge flag">reward-hack flag</span>')
            if revisions is not None:
                badges.append(f'<span class="badge">revisions: {_format_cell(revisions)}</span>')
            code = _escape(str(_lookup(record, "parsed.code") or ""))
            test_summary = _escape(_test_outcome_summary(execution))
            output.append(
                "<details class='rollout'>"
                f"<summary>{task_id} · sample {sample_index} · "
                f"reward {reward} · {difficulty} {''.join(badges)}</summary>"
                f"<pre class='code'>{code}</pre>"
                f"<p class='tests'>{test_summary}</p>"
                "</details>"
            )
    output.append("</section>")
    return output


def _test_outcome_summary(execution: object) -> str:
    if not isinstance(execution, dict):
        return ""
    parts: list[str] = []
    for phase in execution.get("phases", []):
        if not isinstance(phase, dict):
            continue
        outcomes = phase.get("test_outcomes") or []
        if not outcomes:
            continue
        passed = sum(1 for o in outcomes if isinstance(o, dict) and o.get("status") == "passed")
        parts.append(f"{phase.get('name', 'phase')}: {passed}/{len(outcomes)} tests passed")
    return " · ".join(parts) if parts else "no per-test outcomes recorded"


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


def _render_curve_section(curves: Mapping[str, JsonMapping]) -> list[str]:
    rows = []
    series: dict[str, list[tuple[str, float]]] = {
        "Train pass@1": [],
        "Eval pass@1": [],
        "Reward": [],
        "Loss": [],
        "Entropy": [],
        "KL": [],
        "Degenerate": [],
        "Response length": [],
    }
    for label, run in curves.items():
        for point in _list(run, "points"):
            cycle = int(point.get("cycle", 0))
            rows.append(
                [
                    label,
                    cycle,
                    int(point.get("rollout_count", 0)),
                    int(point.get("optimizer_steps", 0)),
                    float(point.get("train_pass_at_1", 0.0)),
                    _optional_number(point.get("eval_pass_at_1")),
                    float(point.get("train_reward_mean", 0.0)),
                    float(point.get("train_loss", 0.0)),
                    float(point.get("policy_loss", 0.0)),
                    float(point.get("value_loss", 0.0)),
                    float(point.get("entropy_loss", 0.0)),
                    float(point.get("kl_loss", 0.0)),
                    float(point.get("mean_approx_kl", 0.0)),
                    float(point.get("train_degenerate_rate", 0.0)),
                    float(point.get("train_response_token_mean", 0.0)),
                ]
            )
            prefix = f"{label} c{cycle}"
            series["Train pass@1"].append((prefix, float(point.get("train_pass_at_1", 0.0))))
            if isinstance(point.get("eval_pass_at_1"), (int, float)):
                series["Eval pass@1"].append((prefix, float(point.get("eval_pass_at_1", 0.0))))
            series["Reward"].append((prefix, float(point.get("train_reward_mean", 0.0))))
            series["Loss"].append((prefix, float(point.get("train_loss", 0.0))))
            series["Entropy"].append((prefix, float(point.get("entropy_loss", 0.0))))
            series["KL"].append((prefix, float(point.get("mean_approx_kl", 0.0))))
            series["Degenerate"].append(
                (prefix, float(point.get("train_degenerate_rate", 0.0)))
            )
            series["Response length"].append(
                (prefix, float(point.get("train_response_token_mean", 0.0)))
            )
    output = [
        '<section class="band">',
        "<h2>Learning Curves</h2>",
        _table(
            (
                "Run",
                "Cycle",
                "Rollouts",
                "Opt Steps",
                "Train pass@1",
                "Eval pass@1",
                "Reward",
                "Loss",
                "Policy",
                "Value",
                "Entropy",
                "KL Loss",
                "Approx KL",
                "Degenerate",
                "Resp Tokens",
            ),
            rows,
            numeric_columns={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        ),
    ]
    for title, values in series.items():
        if not values:
            continue
        floor_upper = (
            1.0
            if title in {"Train pass@1", "Eval pass@1", "Reward", "Degenerate"}
            else 0.0
        )
        output.extend(
            [
                f"<h3>{_escape(title)}</h3>",
                _line_chart(values, floor_upper=floor_upper),
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


def _line_chart(values: list[tuple[str, float]], *, floor_upper: float = 0.0) -> str:
    width = 760
    height = 220
    left = 56
    right = 22
    top = 18
    bottom = 44
    chart_width = width - left - right
    chart_height = height - top - bottom
    raw_values = [value for _, value in values]
    lower = min(0.0, min(raw_values))
    upper = max(0.0, floor_upper, max(raw_values))
    if upper == lower:
        upper = lower + 1.0
    span = upper - lower
    step = chart_width / max(1, len(values) - 1)
    points: list[tuple[float, float]] = []
    for index, (_, value) in enumerate(values):
        x = left + index * step
        y = top + chart_height - ((value - lower) / span * chart_height)
        points.append((x, y))
    zero_y = top + chart_height - ((0.0 - lower) / span * chart_height)
    path = " ".join(
        f"{'M' if index == 0 else 'L'} {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )
    parts = [
        f'<svg class="chart line-chart" viewBox="0 0 {width} {height}" role="img">',
        f'<line class="axis" x1="{left}" y1="{zero_y:.2f}" '
        f'x2="{width - right}" y2="{zero_y:.2f}"></line>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height - bottom}"></line>',
        f'<path class="line-path" d="{path}"></path>',
    ]
    for index, ((label, value), (x, y)) in enumerate(zip(values, points, strict=True)):
        parts.append(f'<circle class="line-point" cx="{x:.2f}" cy="{y:.2f}" r="3"></circle>')
        if index == 0 or index == len(values) - 1:
            parts.append(
                f'<text class="chart-label" x="{x:.2f}" y="{height - 16}">'
                f"{_escape(label)}</text>"
            )
        parts.append(
            f'<text class="chart-value mini" x="{x + 4:.2f}" y="{y - 6:.2f}">'
            f"{value:.3g}</text>"
        )
    parts.append("</svg>")
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
.rollout {
  border: 1px solid var(--line);
  border-radius: 6px;
  margin: 6px 0;
  padding: 6px 10px;
}
.rollout summary {
  cursor: pointer;
  font-size: 13px;
}
.rollout .code {
  background: rgba(127, 127, 127, 0.08);
  border-radius: 4px;
  padding: 8px;
  overflow-x: auto;
  font-size: 12px;
}
.rollout .tests {
  font-size: 12px;
  color: #555;
}
.badge {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 7px;
  border-radius: 10px;
  font-size: 11px;
  background: rgba(127, 127, 127, 0.15);
}
.badge.pass { background: rgba(40, 160, 90, 0.2); }
.badge.fail { background: rgba(200, 70, 70, 0.2); }
.badge.flag { background: rgba(210, 150, 40, 0.25); }
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
.line-path {
  fill: none;
  stroke: var(--accent);
  stroke-width: 2.5;
}
.line-point {
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
.mini {
  font-size: 10px;
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
