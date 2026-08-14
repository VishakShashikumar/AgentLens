"""Renders a self-contained, dependency-free HTML report card from the dict
produced by agentlens/scripts/report_builder.py.

No external CSS/JS/fonts — this has to open correctly from a plain file://
URL on any machine, including one with no internet access, which is also why
every chart here is hand-drawn inline SVG rather than a charting library.
Palette and mark specs follow the project's dataviz skill (see
references/palette.md, marks-and-anatomy.md): fixed categorical order, thin
marks with rounded data-ends, a legend for the two series (v1/before,
v2/after), status color reserved for good/bad deltas with icon + label
(never color alone), light and dark surfaces both defined.
"""

from __future__ import annotations

import html
from pathlib import Path

# --- palette (agentlens dataviz skill reference palette; see palette.md) ---
COLORS = {
    "surface_light": "#fcfcfb",
    "surface_dark": "#1a1a19",
    "page_light": "#f9f9f7",
    "page_dark": "#0d0d0d",
    "text_primary_light": "#0b0b0b",
    "text_primary_dark": "#ffffff",
    "text_secondary_light": "#52514e",
    "text_secondary_dark": "#c3c2b7",
    "muted": "#898781",
    "grid_light": "#e1e0d9",
    "grid_dark": "#2c2c2a",
    "baseline_light": "#c3c2b7",
    "baseline_dark": "#383835",
    "series_v1_light": "#898781",   # baseline/before -> muted, not a hue
    "series_v1_dark": "#6f6d67",
    "series_v2_light": "#2a78d6",   # optimized/after -> categorical slot 1 (blue)
    "series_v2_dark": "#3987e5",
    "good_light": "#0ca30c",
    "good_dark": "#0ca30c",
    "critical_light": "#d03b3b",
    "critical_dark": "#e66767",
    "warning_light": "#fab219",
    "warning_dark": "#fab219",
    "serious_light": "#ec835a",
    "serious_dark": "#ec835a",
}

SEVERITY_COLOR = {"high": "critical", "medium": "warning", "low": "muted"}


def _fmt_int(v: float) -> str:
    return f"{v:,.0f}"


def _fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"


def _esc(s: str) -> str:
    return html.escape(str(s))


def _bar_pair_svg(label: str, unit: str, v1: float, v2: float, fmt=_fmt_int) -> str:
    """One small-multiple: two vertical bars (v1 muted, v2 blue) on their own
    scale, 4px rounded data-end, square baseline, value labels at the tip."""
    w, h = 148, 132
    baseline_y = 104
    max_val = max(v1, v2, 1) * 1.18
    bar_w = 24
    gap = 2
    group_w = bar_w * 2 + gap
    x0 = (w - group_w) / 2
    h1 = (v1 / max_val) * (baseline_y - 18)
    h2 = (v2 / max_val) * (baseline_y - 18)

    def bar(x, bar_h, css_var):
        y = baseline_y - bar_h
        return (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{max(bar_h,1):.1f}" '
            f'rx="4" ry="4" fill="var({css_var})"/>'
            f'<rect x="{x:.1f}" y="{baseline_y-4:.1f}" width="{bar_w}" height="4" fill="var({css_var})"/>'
        )

    label1_y = baseline_y - h1 - 6
    label2_y = baseline_y - h2 - 6
    return f"""
<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{_esc(label)}: v1 {fmt(v1)}, v2 {fmt(v2)}">
  <line x1="8" y1="{baseline_y}" x2="{w-8}" y2="{baseline_y}" stroke="var(--baseline)" stroke-width="1"/>
  {bar(x0, h1, '--series-v1')}
  {bar(x0 + bar_w + gap, h2, '--series-v2')}
  <text x="{x0 + bar_w/2:.1f}" y="{label1_y:.1f}" text-anchor="middle" class="bar-value">{_esc(fmt(v1))}</text>
  <text x="{x0 + bar_w + gap + bar_w/2:.1f}" y="{label2_y:.1f}" text-anchor="middle" class="bar-value">{_esc(fmt(v2))}</text>
  <text x="{w/2}" y="{h-4}" text-anchor="middle" class="axis-label">{_esc(label)}{(' (' + unit + ')') if unit else ''}</text>
</svg>
""".strip()


def _stat_tile(label: str, value_str: str, pct_change: float, good_if_down: bool) -> str:
    is_improvement = (pct_change <= 0) if good_if_down else (pct_change >= 0)
    tone = "good" if is_improvement else "critical"
    arrow = "↓" if pct_change < 0 else ("↑" if pct_change > 0 else "→")
    return f"""
<div class="stat-tile">
  <div class="stat-label">{_esc(label)}</div>
  <div class="stat-value">{_esc(value_str)}</div>
  <div class="stat-delta stat-delta--{tone}"><span aria-hidden="true">{arrow}</span> {_fmt_pct(pct_change)} vs v1</div>
</div>
""".strip()


def _diagnosis_card(d: dict) -> str:
    tone = SEVERITY_COLOR.get(d["severity"], "muted")
    evidence = "".join(f"<li>{_esc(e)}</li>" for e in d["evidence"][:4])
    tasks = ", ".join(d["affected_tasks"][:6]) + (
        f" (+{len(d['affected_tasks'])-6} more)" if len(d["affected_tasks"]) > 6 else ""
    )
    return f"""
<div class="diag-card">
  <div class="diag-head">
    <span class="badge badge--{tone}">{_esc(d['severity'].upper())}</span>
    <span class="diag-pattern">{_esc(d['pattern'])}</span>
    <span class="diag-count">{len(d['affected_tasks'])} task(s)</span>
  </div>
  <p class="diag-explain">{_esc(d['explanation'])}</p>
  <p class="diag-tasks"><span class="muted-label">Affected:</span> {_esc(tasks)}</p>
  <ul class="diag-evidence">{evidence}</ul>
</div>
""".strip()


def _diff_block(diff_text: str) -> str:
    lines = []
    for line in diff_text.splitlines():
        cls = "diff-add" if line.startswith("+") else ("diff-rm" if line.startswith("---") or line.startswith("-") else "diff-ctx")
        lines.append(f'<div class="{cls}">{_esc(line) or "&nbsp;"}</div>')
    return f'<div class="diff-block">{"".join(lines)}</div>'


def _task_table(per_task: list[dict]) -> str:
    rows = []
    for row in per_task:
        ok = "✓" if row["success"] else "✗"
        tone = "good" if row["success"] else "critical"
        rows.append(
            f"<tr><td class='mono'>{_esc(row['task_id'])}</td>"
            f"<td class='status status--{tone}'>{ok}</td>"
            f"<td class='num'>{row['quality_score']}</td>"
            f"<td class='num'>{row['tokens']:,}</td>"
            f"<td class='num'>{row['tool_calls']}</td>"
            f"<td class='num'>{row['latency_ms']}</td>"
            f"<td class='mono small'>{_esc(' → '.join(row['tool_sequence']))}</td></tr>"
        )
    return f"""
<table class="task-table">
  <thead><tr><th>Task</th><th>Pass</th><th>Quality</th><th>Tokens</th><th>Tool calls</th><th>Latency (ms)</th><th>Tool sequence</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>
""".strip()


def _agent_section(report: dict) -> str:
    v1, v2, deltas = report["v1"], report["v2"], report["deltas"]
    charts = "".join(
        [
            f'<div class="chart-cell">{_bar_pair_svg("Total tokens", "", v1["total_tokens"], v2["total_tokens"])}</div>',
            f'<div class="chart-cell">{_bar_pair_svg("Tool calls", "", v1["total_tool_calls"], v2["total_tool_calls"])}</div>',
            f'<div class="chart-cell">{_bar_pair_svg("Avg latency", "ms", v1["avg_latency_ms"], v2["avg_latency_ms"], fmt=lambda v: f"{v:.1f}" if v < 100 else f"{v:.0f}")}</div>',
        ]
    )
    latency_val_str = f"{v2['avg_latency_ms']:.1f} ms" if v2["avg_latency_ms"] < 100 else f"{v2['avg_latency_ms']:.0f} ms"
    tiles = "".join(
        [
            _stat_tile("Total tokens", f"{v2['total_tokens']:,}", deltas["total_tokens_pct_change"], good_if_down=True),
            _stat_tile("Tool calls", f"{v2['total_tool_calls']:,}", deltas["total_tool_calls_pct_change"], good_if_down=True),
            _stat_tile("Avg latency", latency_val_str, deltas["avg_latency_pct_change"], good_if_down=True),
            _stat_tile("Quality score", f"{v2['avg_quality_score']:.2f}", deltas["avg_quality_score_delta"] * 100, good_if_down=False),
        ]
    )
    diagnosis_html = "".join(_diagnosis_card(d) for d in report["diagnosis"]) or "<p class='muted-label'>No failure patterns detected.</p>"
    diff_html = _diff_block(report["diff"])
    table_html = _task_table(v2["per_task"])

    return f"""
<section class="agent-section">
  <h2>{_esc(report['agent_name'])}</h2>
  <div class="legend-row">
    <span class="legend-swatch legend-swatch--v1"></span> v1 (baseline)
    &nbsp;&nbsp;
    <span class="legend-swatch legend-swatch--v2"></span> v2 (optimized)
  </div>
  <div class="chart-row">{charts}</div>
  <div class="stat-row">{tiles}</div>

  <h3>Diagnosis (v1)</h3>
  <div class="diag-grid">{diagnosis_html}</div>

  <h3>Fix applied</h3>
  {diff_html}

  <h3>Per-task results (v2)</h3>
  {table_html}
</section>
""".strip()


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentLens Report Card</title>
<style>
  :root {{
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7;
    --series-v1: #898781; --series-v2: #2a78d6;
    --good: #0ca30c; --critical: #d03b3b; --warning: #fab219; --serious: #ec835a;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835;
      --series-v1: #6f6d67; --series-v2: #3987e5;
      --good: #0ca30c; --critical: #e66767; --warning: #fab219; --serious: #ec835a;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 80px; background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  header.hero {{ margin-bottom: 8px; }}
  header.hero h1 {{ font-size: 28px; margin: 0 0 4px; }}
  header.hero p {{ color: var(--text-secondary); margin: 0 0 24px; font-size: 14px; }}
  h2 {{ font-size: 20px; margin: 0 0 4px; }}
  h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-secondary);
        margin: 28px 0 10px; }}
  section.agent-section {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 12px;
    padding: 24px 28px; margin-bottom: 28px;
  }}
  .legend-row {{ font-size: 13px; color: var(--text-secondary); margin: 4px 0 18px; }}
  .legend-swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; }}
  .legend-swatch--v1 {{ background: var(--series-v1); }}
  .legend-swatch--v2 {{ background: var(--series-v2); }}
  .chart-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .chart-cell svg text.bar-value {{ font-size: 10px; fill: var(--text-secondary); font-variant-numeric: tabular-nums; }}
  .chart-cell svg text.axis-label {{ font-size: 10px; fill: var(--muted); }}
  .stat-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 20px; }}
  .stat-tile {{ border: 1px solid var(--grid); border-radius: 10px; padding: 12px 14px; }}
  .stat-label {{ font-size: 12px; color: var(--text-secondary); }}
  .stat-value {{ font-size: 22px; font-weight: 600; margin: 2px 0; }}
  .stat-delta {{ font-size: 12px; font-weight: 600; }}
  .stat-delta--good {{ color: var(--good); }}
  .stat-delta--critical {{ color: var(--critical); }}
  .diag-grid {{ display: flex; flex-direction: column; gap: 10px; }}
  .diag-card {{ border: 1px solid var(--grid); border-radius: 10px; padding: 12px 16px; }}
  .diag-head {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
  .diag-pattern {{ font-weight: 600; }}
  .diag-count {{ color: var(--muted); font-size: 12px; margin-left: auto; }}
  .diag-explain {{ margin: 4px 0; font-size: 13px; color: var(--text-secondary); }}
  .diag-tasks {{ font-size: 12px; margin: 4px 0; }}
  .diag-evidence {{ margin: 6px 0 0; padding-left: 18px; font-size: 12px; color: var(--text-secondary); font-family: ui-monospace, monospace; }}
  .muted-label {{ color: var(--muted); }}
  .badge {{ font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 4px; color: #fff; }}
  .badge--critical {{ background: var(--critical); }}
  .badge--warning {{ background: var(--warning); color: #2a2000; }}
  .badge--muted {{ background: var(--muted); }}
  .diff-block {{ background: var(--page); border: 1px solid var(--grid); border-radius: 8px; padding: 10px 14px;
                 font-family: ui-monospace, "SF Mono", monospace; font-size: 12.5px; overflow-x: auto; }}
  .diff-add {{ color: var(--good); }}
  .diff-rm {{ color: var(--text-secondary); }}
  .diff-ctx {{ color: var(--text-secondary); }}
  table.task-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  table.task-table th {{ text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
                          text-transform: uppercase; letter-spacing: .03em; border-bottom: 1px solid var(--grid);
                          padding: 6px 8px; }}
  table.task-table td {{ padding: 6px 8px; border-bottom: 1px solid var(--grid); }}
  table.task-table td.num {{ font-variant-numeric: tabular-nums; text-align: right; }}
  table.task-table td.mono {{ font-family: ui-monospace, monospace; }}
  table.task-table td.small {{ font-size: 11px; color: var(--text-secondary); }}
  .status--good {{ color: var(--good); font-weight: 700; }}
  .status--critical {{ color: var(--critical); font-weight: 700; }}
  footer {{ max-width: 1000px; margin: 24px auto 0; color: var(--muted); font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>AgentLens Report Card</h1>
    <p>Baseline (v1) vs. AgentLens-optimized (v2) — {provider_note}</p>
  </header>
  {sections}
</div>
<footer>Generated by AgentLens · agentlens/scripts/run_audit.py</footer>
</body>
</html>
"""


def generate_html_report(research_report: dict, coding_report: dict, out_path: Path) -> Path:
    sections = _agent_section(research_report) + "\n" + _agent_section(coding_report)
    html_doc = PAGE_TEMPLATE.format(
        sections=sections,
        provider_note="figures below are from a real run of this repo's own eval harness",
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    return out_path
