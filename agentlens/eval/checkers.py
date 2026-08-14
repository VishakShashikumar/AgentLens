"""Rule-based checks over traces.

These are generic, task-agnostic facts about *how* an agent behaved — they
don't know what "correct" looks like for a given task (that's the judge's
job); they know what "wasteful" looks like for any task. agentlens/diagnosis
turns these raw findings into a named, explained pattern.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from agentlens.tracing.tracer import Trace


@dataclass
class RuleFinding:
    task_id: str
    rule: str
    detail: str
    steps: tuple[int, ...]


def _args_similar(a: dict | None, b: dict | None) -> bool:
    if a is None or b is None:
        return a == b
    if json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True):
        return True
    for va, vb in zip(a.values(), b.values()):
        if isinstance(va, str) and isinstance(vb, str) and va and vb:
            shorter, longer = (va, vb) if len(va) <= len(vb) else (vb, va)
            if longer.startswith(shorter):
                return True
    return False


def check_redundant_tool_calls(trace: Trace) -> list[RuleFinding]:
    """Same tool called twice with the same or near-identical arguments."""
    calls = trace.tool_calls()
    findings = []
    flagged_pairs: set[tuple[int, int]] = set()
    for i in range(len(calls)):
        for j in range(i + 1, len(calls)):
            if calls[i].name != calls[j].name:
                continue
            if _args_similar(calls[i].args, calls[j].args) and (i, j) not in flagged_pairs:
                findings.append(
                    RuleFinding(
                        task_id=trace.task_id,
                        rule="redundant_tool_calls",
                        detail=f"'{calls[i].name}' called again with near-identical arguments "
                        f"(steps {i} and {j})",
                        steps=(i, j),
                    )
                )
                flagged_pairs.add((i, j))
    return findings


def check_excess_same_tool_calls(trace: Trace, tool_name: str, threshold: int = 3) -> list[RuleFinding]:
    """A single tool called an unusually large number of times in one run."""
    idxs = [i for i, s in enumerate(trace.tool_calls()) if s.name == tool_name]
    if len(idxs) >= threshold:
        return [
            RuleFinding(
                task_id=trace.task_id,
                rule="excess_same_tool_calls",
                detail=f"'{tool_name}' called {len(idxs)} times in a single run",
                steps=tuple(idxs),
            )
        ]
    return []


def check_token_outlier(traces: list[Trace], z: float = 0.4) -> list[RuleFinding]:
    """Traces whose token usage is well above the batch median (batch-level check)."""
    if len(traces) < 2:
        return []
    totals = [t.total_tokens for t in traces]
    median = statistics.median(totals)
    findings = []
    for t in traces:
        if median > 0 and t.total_tokens >= median * (1 + z):
            findings.append(
                RuleFinding(
                    task_id=t.task_id,
                    rule="token_outlier",
                    detail=f"{t.total_tokens} tokens vs. batch median {median:.0f}",
                    steps=(),
                )
            )
    return findings


def run_all_checks(traces: list[Trace]) -> list[RuleFinding]:
    findings: list[RuleFinding] = []
    for t in traces:
        findings += check_redundant_tool_calls(t)
        findings += check_excess_same_tool_calls(t, "run_tests", threshold=3)
        findings += check_excess_same_tool_calls(t, "read_file", threshold=2)
    findings += check_token_outlier(traces)
    return findings
