"""The research agent — and its mock policy.

The mock policy is intentionally not "smart": it follows each task's known
correct tool-call plan. The interesting part is that it also *misbehaves* in
specific, named ways whenever the current system prompt is missing the fix
for that misbehavior — see PROMPT_FIXES. That's what gives AgentLens
something real to diagnose and something real to fix.
"""

from __future__ import annotations

from agentlens.agents.base import AgentConfig, run_agent_task
from agentlens.eval.tasks import ResearchTask
from agentlens.llm.client import LLMClient
from agentlens.tools.research_tools import RESEARCH_TOOL_IMPL, RESEARCH_TOOL_SPECS
from agentlens.tracing.tracer import Trace

AGENT_NAME = "research_agent"

# These are real, well-formed prompt-engineering instructions — not magic
# strings. A real Claude model reading them would behave better too; the
# mock policy just also uses their presence/absence as its behavior switch,
# so the same fix works whether the agent behind it is scripted or real.
PROMPT_FIXES = {
    "dedupe": (
        "Before calling a tool, check whether you already have this information "
        "from an earlier step in this conversation; do not repeat identical or "
        "near-identical calls."
    ),
    "concise": (
        "Keep your reasoning concise: do not restate full tool outputs, and do not "
        "pad responses with unnecessary explanation."
    ),
}

BASE_SYSTEM_PROMPT = (
    "You are a research assistant. You have access to web_search, calculator, and "
    "save_note tools. Use them to answer the user's question accurately, then give a "
    "final answer in one or two sentences."
)


def default_config(version: str = "v1") -> AgentConfig:
    return AgentConfig(version=version, system_prompt=BASE_SYSTEM_PROMPT, max_steps=8)


def _reasoning_text(tool_name: str, args: dict, concise: bool) -> str:
    if concise:
        return f"Calling {tool_name}."
    return (
        f"I need to find information to answer this question, so let me think through "
        f"this carefully, one step at a time, before doing anything else. First, I will "
        f"call the {tool_name} tool with arguments {args!r}, since that should give me "
        f"the information I need in order to make progress toward a complete and "
        f"well-supported final answer."
    )


def _collect_tool_results(messages: list[dict]) -> list[str]:
    results = []
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append(block["content"])
    return results


def make_research_policy(task: ResearchTask, system_prompt: str):
    dedupe_on = PROMPT_FIXES["dedupe"] in system_prompt
    concise_on = PROMPT_FIXES["concise"] in system_prompt

    canonical_plan = list(task.ideal_plan)
    executed_plan = list(canonical_plan)
    dup_index = None

    if "duplicate_search" in task.inefficiency and not dedupe_on:
        for i, (name, args) in enumerate(canonical_plan):
            if name == "web_search":
                dup_index = i + 1
                dup_call = ("web_search", {"query": args["query"] + " - double-check"})
                executed_plan = canonical_plan[: i + 1] + [dup_call] + canonical_plan[i + 1 :]
                break

    call_count = {"n": 0}

    def policy_fn(system: str, messages: list[dict], tools: list[dict]):
        idx = call_count["n"]
        if idx < len(executed_plan):
            name, args = executed_plan[idx]
            call_count["n"] += 1
            reasoning = _reasoning_text(name, args, concise_on)
            return reasoning, [(name, args)]

        results = _collect_tool_results(messages)
        canonical_results = [r for i, r in enumerate(results) if i != dup_index]
        answer = task.compose_answer(canonical_results)
        return answer, []

    return policy_fn


def _dispatch(name: str, args: dict) -> str:
    return RESEARCH_TOOL_IMPL[name](**args)


def run_task(llm_client: LLMClient, config: AgentConfig, task: ResearchTask) -> Trace:
    policy_fn = make_research_policy(task, config.system_prompt)
    return run_agent_task(
        llm_client=llm_client,
        agent_name=AGENT_NAME,
        config=config,
        tool_specs=RESEARCH_TOOL_SPECS,
        dispatch_fn=_dispatch,
        policy_fn=policy_fn,
        task_id=task.id,
        initial_prompt=task.prompt,
    )
