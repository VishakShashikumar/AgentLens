"""Turns a v1/v2 audit run into a plain-dict report — the single artifact
consumed by both the console summary and the HTML report card."""

from __future__ import annotations

from agentlens.agents.base import AgentConfig
from agentlens.diagnosis.diagnoser import DiagnosisReport
from agentlens.eval.harness import EvalResult
from agentlens.optimizer.optimizer_agent import PatchResult


def pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return (after - before) / before * 100.0


def _per_task(ev: EvalResult) -> list[dict]:
    rows = []
    for trace in ev.traces:
        verdict = ev.verdict_for(trace.task_id)
        rows.append(
            {
                "task_id": trace.task_id,
                "success": verdict.task_success if verdict else None,
                "quality_score": round(verdict.quality_score, 2) if verdict else None,
                "tokens": trace.total_tokens,
                "tool_calls": trace.tool_call_count,
                "tool_sequence": [s.name for s in trace.tool_calls()],
                "latency_ms": round(trace.total_latency_ms, 1),
            }
        )
    return rows


def _eval_summary(ev: EvalResult) -> dict:
    return {
        "config_version": ev.config_version,
        "n_tasks": ev.n_tasks,
        "success_rate": round(ev.success_rate, 3),
        "avg_quality_score": round(ev.avg_quality_score, 3),
        "total_tokens": ev.total_agent_tokens,
        "avg_tokens_per_task": round(ev.avg_agent_tokens, 1),
        "total_tool_calls": ev.total_tool_calls,
        "avg_latency_ms": round(ev.avg_latency_ms, 2),
        "judge_tokens": ev.judge_tokens,
        "per_task": _per_task(ev),
    }


def build_agent_report(
    agent_name: str,
    v1: EvalResult,
    v2: EvalResult,
    diag_report: DiagnosisReport,
    patch: PatchResult,
    v1_config: AgentConfig,
    v2_config: AgentConfig,
) -> dict:
    return {
        "agent_name": agent_name,
        "v1": _eval_summary(v1),
        "v2": _eval_summary(v2),
        "diagnosis": [
            {
                "pattern": d.pattern,
                "severity": d.severity,
                "affected_tasks": d.affected_tasks,
                "explanation": d.explanation,
                "evidence": d.evidence,
                "recommended_fix_key": d.recommended_fix_key,
            }
            for d in diag_report.diagnoses
        ],
        "applied_fixes": patch.applied_fixes,
        "diff": patch.diff,
        "v1_system_prompt": v1_config.system_prompt,
        "v2_system_prompt": v2_config.system_prompt,
        "deltas": {
            "success_rate_pp": round((v2.success_rate - v1.success_rate) * 100, 1),
            "avg_quality_score_delta": round(v2.avg_quality_score - v1.avg_quality_score, 3),
            "total_tokens_pct_change": round(pct_change(v1.total_agent_tokens, v2.total_agent_tokens), 1),
            "total_tool_calls_pct_change": round(pct_change(v1.total_tool_calls, v2.total_tool_calls), 1),
            "avg_latency_pct_change": round(pct_change(v1.avg_latency_ms, v2.avg_latency_ms), 1),
        },
    }
