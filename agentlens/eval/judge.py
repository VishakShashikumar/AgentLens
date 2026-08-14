"""Task success grading + an LLM-as-judge efficiency score.

Two separate signals, deliberately kept apart:

- ``task_success`` is graded programmatically wherever the task has a
  checkable ground truth (a required substring, a passing test). Using a
  program instead of an LLM wherever a program *can* judge correctness is
  itself an evaluation-design decision worth being able to defend in an
  interview — LLM-judges are for open-ended quality, not for facts you can
  check directly.
- ``quality_score`` comes from an actual LLM-judge call (routed through the
  same LLMClient the agents use, so it shows up as its own traced, costed
  call) that grades efficiency: did the agent get there without wasted
  steps. This is the score the optimizer is ultimately trying to improve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agentlens.eval.tasks import CodingTask, ResearchTask
from agentlens.llm.client import LLMClient, Usage
from agentlens.tracing.tracer import Trace

JUDGE_SYSTEM_PROMPT = (
    "You are grading an AI agent's execution trace for efficiency. You will see how "
    "many tool calls it made, how many of those were redundant, and whether the task "
    "succeeded. Score quality from 0 to 1: reward correct, efficient completions; "
    "penalize redundant tool calls and unnecessary verbosity. Respond with the score "
    "and a one-sentence rationale."
)


@dataclass
class JudgeVerdict:
    task_id: str
    task_success: bool
    quality_score: float
    rationale: str


def _programmatic_success(trace: Trace, task, task_kind: str) -> bool:
    if task_kind == "research":
        answer = (trace.final_answer or "").lower()
        return all(sub.lower() in answer for sub in task.answer_contains)
    if task_kind == "coding":
        for span in trace.tool_calls():
            if span.name == "run_tests" and span.result_repr and span.result_repr.startswith("PASSED"):
                return True
        return False
    raise ValueError(f"Unknown task_kind: {task_kind}")


def _judge_policy_fn(trace: Trace, task_success: bool, ideal_call_count: int):
    redundant = max(0, trace.tool_call_count - ideal_call_count)

    def policy_fn(system: str, messages: list[dict], tools: list[dict]):
        score = 1.0 if task_success else 0.2
        score -= 0.15 * redundant
        score = max(0.0, min(1.0, score))
        rationale = (
            f"Task {'succeeded' if task_success else 'failed'} in {trace.tool_call_count} "
            f"tool call(s) against an ideal of {ideal_call_count} — "
            f"{redundant} call(s) judged redundant."
        )
        return f"{score:.2f} | {rationale}", []

    return policy_fn


def judge_trace(
    llm_client: LLMClient,
    trace: Trace,
    task,
    task_kind: str,
    ideal_call_count: int,
) -> tuple[JudgeVerdict, Usage]:
    task_success = _programmatic_success(trace, task, task_kind)
    policy_fn = _judge_policy_fn(trace, task_success, ideal_call_count)

    transcript_summary = (
        f"Task: {task.prompt}\nTool calls made: {trace.tool_call_count}\n"
        f"Ideal tool calls: {ideal_call_count}\nSucceeded: {task_success}"
    )
    resp = llm_client.generate(
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript_summary}],
        tools=[],
        policy_fn=policy_fn,
    )
    score_str, _, rationale = (resp.text or "0.0 | no rationale").partition(" | ")
    try:
        score = float(score_str)
    except ValueError:
        score = 0.0

    verdict = JudgeVerdict(
        task_id=trace.task_id, task_success=task_success, quality_score=score, rationale=rationale
    )
    return verdict, resp.usage
