"""Optional interactive Streamlit dashboard — loads the latest JSON report
written by scripts/run_audit.py and lets you click through tasks live.

The static HTML report (report_html.py) is the primary, dependency-free
deliverable; this is for local, interactive exploration once you have
streamlit installed (`pip install streamlit`, then `streamlit run
agentlens/dashboard/app.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def _latest_report() -> dict | None:
    candidates = sorted(REPORTS_DIR.glob("audit_*.json"), reverse=True)
    if not candidates:
        return None
    return json.loads(candidates[0].read_text())


def render_agent(name: str, report: dict) -> None:
    st.header(name)
    v1, v2, d = report["v1"], report["v2"], report["deltas"]

    cols = st.columns(4)
    cols[0].metric("Total tokens", f"{v2['total_tokens']:,}", f"{d['total_tokens_pct_change']:.1f}%")
    cols[1].metric("Tool calls", v2["total_tool_calls"], f"{d['total_tool_calls_pct_change']:.1f}%")
    cols[2].metric("Avg latency (ms)", f"{v2['avg_latency_ms']:.1f}", f"{d['avg_latency_pct_change']:.1f}%")
    cols[3].metric("Quality score", f"{v2['avg_quality_score']:.2f}", f"{d['avg_quality_score_delta']:.2f}")

    st.subheader("Diagnosis (v1)")
    if not report["diagnosis"]:
        st.write("No failure patterns detected.")
    for diag in report["diagnosis"]:
        with st.expander(f"[{diag['severity'].upper()}] {diag['pattern']} — {len(diag['affected_tasks'])} task(s)"):
            st.write(diag["explanation"])
            st.write("Affected:", ", ".join(diag["affected_tasks"]))
            st.code("\n".join(diag["evidence"]))

    st.subheader("Fix applied")
    st.code(report["diff"], language="diff")

    st.subheader("Per-task results (v2)")
    st.dataframe(v2["per_task"], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="AgentLens", layout="wide")
    st.title("AgentLens — Agent Auditor & Optimizer")
    report = _latest_report()
    if report is None:
        st.warning("No audit report found yet. Run: `python -m agentlens.scripts.run_audit`")
        return
    render_agent("research_agent", report["research_agent"])
    render_agent("coding_agent", report["coding_agent"])


if __name__ == "__main__":
    main()
