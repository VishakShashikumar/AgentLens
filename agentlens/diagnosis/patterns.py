"""A small field guide to agent failure patterns.

Each pattern turns raw rule findings (agentlens/eval/checkers.py) into a
named, explained diagnosis with a concrete recommended fix. Naming these
things is not decoration — "Tool-Call Ping-Pong" is a label a teammate can
say out loud in a standup and everyone knows what you mean, the same way
"N+1 query" or "thundering herd" work. That's the point of this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agentlens.eval.checkers import RuleFinding
from agentlens.eval.harness import EvalResult

# Tool names that, when duplicated, mean "the agent asked the world the same
# question twice" rather than "the agent asked two different questions."
_QUERY_TOOLS = {"web_search"}
_READ_TOOLS = {"read_file"}
_VERIFY_TOOLS = {"run_tests"}


@dataclass
class Diagnosis:
    pattern: str
    severity: str  # "high" | "medium" | "low"
    affected_tasks: list[str]
    evidence: list[str]
    explanation: str
    recommended_fix_key: str  # key into agentlens.optimizer.optimizer_agent.FIX_LIBRARY

    @property
    def impact_score(self) -> float:
        weight = {"high": 3, "medium": 2, "low": 1}[self.severity]
        return weight * len(self.affected_tasks)


def _findings_by_rule(findings: list[RuleFinding], rule: str) -> list[RuleFinding]:
    return [f for f in findings if f.rule == rule]


def detect_tool_call_ping_pong(findings: list[RuleFinding]) -> Diagnosis | None:
    """The agent calls the same lookup tool twice for (near) the same query —
    it doesn't trust or remember its own prior result."""
    hits = [
        f
        for f in _findings_by_rule(findings, "redundant_tool_calls")
        if any(tool in f.detail for tool in _QUERY_TOOLS)
    ]
    if not hits:
        return None
    tasks = sorted({f.task_id for f in hits})
    return Diagnosis(
        pattern="Tool-Call Ping-Pong",
        severity="medium",
        affected_tasks=tasks,
        evidence=[f.detail for f in hits],
        explanation=(
            "The agent re-issues a near-identical search instead of reusing the result "
            "it already has in context. Each occurrence costs one extra tool round-trip "
            "and its full token overhead for zero new information."
        ),
        recommended_fix_key="dedupe_tool_calls",
    )


def detect_context_rot(findings: list[RuleFinding]) -> Diagnosis | None:
    """The agent re-reads a source it already loaded instead of reusing it
    from context — a sign it isn't tracking what it already knows."""
    hits = [
        f
        for f in _findings_by_rule(findings, "excess_same_tool_calls")
        if any(tool in f.detail for tool in _READ_TOOLS)
    ] + [
        f
        for f in _findings_by_rule(findings, "redundant_tool_calls")
        if any(tool in f.detail for tool in _READ_TOOLS)
    ]
    if not hits:
        return None
    tasks = sorted({f.task_id for f in hits})
    return Diagnosis(
        pattern="Context Rot",
        severity="medium",
        affected_tasks=tasks,
        evidence=[f.detail for f in hits],
        explanation=(
            "The agent reloads a file it already read in this same run rather than "
            "reusing the content sitting earlier in its own context window."
        ),
        recommended_fix_key="reuse_context",
    )


def detect_retry_storm(findings: list[RuleFinding]) -> Diagnosis | None:
    """The agent keeps re-verifying after it already has a passing result —
    it doesn't know when to stop."""
    hits = [
        f
        for f in _findings_by_rule(findings, "excess_same_tool_calls")
        if any(tool in f.detail for tool in _VERIFY_TOOLS)
    ]
    if not hits:
        return None
    tasks = sorted({f.task_id for f in hits})
    return Diagnosis(
        pattern="Retry Storm",
        severity="high",
        affected_tasks=tasks,
        evidence=[f.detail for f in hits],
        explanation=(
            "The agent re-runs verification (tests) multiple times after already "
            "succeeding, instead of stopping at the first passing result. In a real "
            "CI-integrated agent this is the pattern that quietly triples your bill."
        ),
        recommended_fix_key="stop_on_success",
    )


def detect_prompt_bloat(eval_result: EvalResult, token_threshold: float = 20.0) -> Diagnosis | None:
    """A large share of the agent's own *reasoning text* is padding — restating
    context instead of moving the task forward. Deliberately measured from the
    text alone (not tool-call payloads — writing a large file is legitimately
    token-heavy and shouldn't be confused with wasteful reasoning)."""
    from agentlens.llm.client import estimate_tokens

    over_threshold: dict[str, float] = {}
    for trace in eval_result.traces:
        llm_spans = [s for s in trace.spans if s.kind == "llm_call"]
        if not llm_spans:
            continue
        reasoning_tokens = [estimate_tokens(s.result_repr or "") for s in llm_spans]
        avg_reasoning = sum(reasoning_tokens) / len(reasoning_tokens)
        # A terse "Calling web_search." is ~5 tokens; anything routinely well
        # above that on a simple step is padding, not reasoning.
        if avg_reasoning > token_threshold:
            over_threshold[trace.task_id] = avg_reasoning
    if not over_threshold:
        return None
    return Diagnosis(
        pattern="Prompt Bloat",
        severity="high",
        affected_tasks=sorted(over_threshold),
        evidence=[f"{task_id}: avg {round(v)} reasoning tokens/step" for task_id, v in over_threshold.items()][:5],
        explanation=(
            "The agent's reasoning text before each tool call restates the task and its "
            "own plan at length instead of stating the action tersely. This inflates "
            "output tokens on every single step, not just the wasteful ones — it's the "
            "most expensive pattern per occurrence even though each instance looks small."
        ),
        recommended_fix_key="concise_reasoning",
    )


def run_all_detectors(eval_result: EvalResult) -> list[Diagnosis]:
    diagnoses = [
        detect_tool_call_ping_pong(eval_result.rule_findings),
        detect_context_rot(eval_result.rule_findings),
        detect_retry_storm(eval_result.rule_findings),
        detect_prompt_bloat(eval_result),
    ]
    found = [d for d in diagnoses if d is not None]
    return sorted(found, key=lambda d: d.impact_score, reverse=True)
