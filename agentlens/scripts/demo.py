"""A plain-English, narrated walkthrough of AgentLens — run this to *watch*
what the tool does, step by step, instead of just reading a table of
numbers. This is the one to run live in an interview.

Usage:
    python -m agentlens.scripts.demo
"""

from __future__ import annotations

from agentlens.agents import research_agent
from agentlens.eval.checkers import check_redundant_tool_calls, check_excess_same_tool_calls
from agentlens.eval.tasks import RESEARCH_TASKS
from agentlens.llm.client import get_client
from agentlens.tracing.tracer import Trace


def _shorten(text: str | None, n: int = 70) -> str:
    if not text:
        return ""
    return text if len(text) <= n else text[: n - 1] + "…"


def narrate_trace(trace: Trace) -> None:
    step_num = 1
    for span in trace.spans:
        if span.kind != "tool_call":
            continue
        print(f"    Step {step_num}: the agent calls `{span.name}` with {span.args}")
        print(f"             → gets back: \"{_shorten(span.result_repr)}\"")
        step_num += 1
    print(f'    Final answer: "{trace.final_answer}"')

    findings = check_redundant_tool_calls(trace) + check_excess_same_tool_calls(trace, "web_search", threshold=3)
    if findings:
        print("    ⚠ Wasteful — this agent repeated a step it didn't need to:")
        for f in findings:
            print(f"        - {f.detail}")
    else:
        print("    ✓ No wasted steps — every tool call was necessary.")

    print(f"    Totals: {trace.tool_call_count} tool call(s), {trace.total_tokens} tokens used.\n")


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # fine in mock mode; needed only to auto-load ANTHROPIC_API_KEY from .env

    client = get_client()
    task = next(t for t in RESEARCH_TASKS if t.id == "r2_eu_states_per_capita")

    print("=" * 72)
    print(f"AGENTLENS DEMO — provider: {type(client).__name__}")
    print("same task, run twice: before and after the fix")
    print("=" * 72)
    print(f'\nThe task we\'re giving the agent: "{task.prompt}"\n')

    print("-" * 72)
    print("RUN 1 — the agent's ORIGINAL instructions (v1, unfixed)")
    print("-" * 72)
    v1_config = research_agent.default_config("v1")
    v1_trace = research_agent.run_task(client, v1_config, task)
    narrate_trace(v1_trace)

    print("-" * 72)
    print("What AgentLens does in between: it looks at that run, recognizes the")
    print("repeated search as the 'Tool-Call Ping-Pong' pattern, and adds one line")
    print("to the agent's instructions telling it to check its own notes before")
    print("searching again.")
    print("-" * 72 + "\n")

    print("-" * 72)
    print("RUN 2 — the agent's FIXED instructions (v2, after AgentLens's patch)")
    print("-" * 72)
    v2_config = research_agent.default_config("v2")
    v2_config.system_prompt += "\n" + research_agent.PROMPT_FIXES["dedupe"]
    v2_trace = research_agent.run_task(client, v2_config, task)
    narrate_trace(v2_trace)

    saved_calls = v1_trace.tool_call_count - v2_trace.tool_call_count
    saved_tokens = v1_trace.total_tokens - v2_trace.total_tokens
    print("=" * 72)
    print(
        f"RESULT: same task, same correct answer, but {saved_calls} fewer tool call(s) "
        f"and {saved_tokens} fewer tokens spent — just from one added instruction."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
