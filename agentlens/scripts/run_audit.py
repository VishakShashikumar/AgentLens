"""AgentLens end-to-end audit: baseline eval -> diagnose -> optimize ->
re-eval -> report. This is the command a resume demo / interview walkthrough
runs.

Usage:
    python -m agentlens.scripts.run_audit
    AGENTLENS_LLM_PROVIDER=anthropic python -m agentlens.scripts.run_audit
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agentlens.agents import coding_agent, research_agent
from agentlens.diagnosis.diagnoser import diagnose
from agentlens.eval import harness
from agentlens.llm.client import get_client
from agentlens.optimizer.optimizer_agent import propose_patch
from agentlens.scripts.report_builder import build_agent_report

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"


def audit_agent(agent_name: str, llm_client, default_config_fn, run_eval_fn) -> dict:
    v1_config = default_config_fn("v1")
    v1_result = run_eval_fn(llm_client, v1_config)

    diag_report = diagnose(v1_result)

    patch = propose_patch(v1_config, diag_report, new_version="v2")
    v2_result = run_eval_fn(llm_client, patch.new_config)

    return build_agent_report(
        agent_name=agent_name,
        v1=v1_result,
        v2=v2_result,
        diag_report=diag_report,
        patch=patch,
        v1_config=v1_config,
        v2_config=patch.new_config,
    )


def print_summary(report: dict) -> None:
    v1, v2, d = report["v1"], report["v2"], report["deltas"]
    print(f"\n{'=' * 60}\n{report['agent_name']}\n{'=' * 60}")
    print(f"{'metric':<22}{'v1':>10}{'v2':>10}{'change':>14}")
    print(f"{'success rate':<22}{v1['success_rate']*100:>9.0f}%{v2['success_rate']*100:>9.0f}%"
          f"{d['success_rate_pp']:>+13.1f}pp")
    print(f"{'avg quality score':<22}{v1['avg_quality_score']:>10.2f}{v2['avg_quality_score']:>10.2f}"
          f"{d['avg_quality_score_delta']:>+14.2f}")
    print(f"{'total tokens':<22}{v1['total_tokens']:>10}{v2['total_tokens']:>10}"
          f"{d['total_tokens_pct_change']:>+13.1f}%")
    print(f"{'total tool calls':<22}{v1['total_tool_calls']:>10}{v2['total_tool_calls']:>10}"
          f"{d['total_tool_calls_pct_change']:>+13.1f}%")
    print(f"{'avg latency (ms)':<22}{v1['avg_latency_ms']:>10.1f}{v2['avg_latency_ms']:>10.1f}"
          f"{d['avg_latency_pct_change']:>+13.1f}%")

    if report["diagnosis"]:
        print("\nDiagnosis (v1):")
        for d_item in report["diagnosis"]:
            print(f"  [{d_item['severity'].upper()}] {d_item['pattern']} "
                  f"— {len(d_item['affected_tasks'])} task(s)")
    print(f"\nFixes applied for v2: {', '.join(report['applied_fixes']) or '(none)'}")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # fine in mock mode; needed only to auto-load ANTHROPIC_API_KEY from .env

    llm_client = get_client()
    provider = type(llm_client).__name__
    print(f"AgentLens audit — provider: {provider}")

    research_report = audit_agent(
        "research_agent", llm_client, research_agent.default_config, harness.run_research_eval
    )
    coding_report = audit_agent(
        "coding_agent", llm_client, coding_agent.default_config, harness.run_coding_eval
    )

    print_summary(research_report)
    print_summary(coding_report)

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"audit_{int(time.time())}.json"
    with out_path.open("w") as f:
        json.dump({"research_agent": research_report, "coding_agent": coding_report}, f, indent=2)
    print(f"\nFull report written to {out_path}")

    try:
        from agentlens.dashboard.report_html import generate_html_report

        html_path = REPORTS_DIR / "latest_report.html"
        generate_html_report(research_report, coding_report, html_path)
        print(f"HTML report card written to {html_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
