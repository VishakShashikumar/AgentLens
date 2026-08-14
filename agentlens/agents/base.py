"""Generic tool-calling agent loop, shared by every target agent.

This is the one piece of code that both the mock provider and a real
Anthropic-backed run execute identically — which is the whole point of the
LLMClient abstraction: the loop doesn't know or care whether ``generate()``
came from a script or a real model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from agentlens.llm.client import LLMClient
from agentlens.tracing.tracer import Trace, Tracer


@dataclass
class AgentConfig:
    version: str
    system_prompt: str
    max_steps: int = 8


def run_agent_task(
    *,
    llm_client: LLMClient,
    agent_name: str,
    config: AgentConfig,
    tool_specs: list[dict],
    dispatch_fn: Callable[[str, dict], str],
    policy_fn: Optional[Callable],
    task_id: str,
    initial_prompt: str,
) -> Trace:
    tracer = Tracer(agent_name=agent_name, config_version=config.version, task_id=task_id)
    messages: list[dict] = [{"role": "user", "content": initial_prompt}]

    final_text: Optional[str] = None
    for _step in range(config.max_steps):
        t0 = time.perf_counter()
        resp = llm_client.generate(
            system=config.system_prompt,
            messages=messages,
            tools=tool_specs,
            policy_fn=policy_fn,
        )
        t1 = time.perf_counter()
        tracer.record(
            kind="llm_call",
            name="assistant_turn",
            started_at=t0,
            ended_at=t1,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            result_repr=resp.text,
        )

        if not resp.tool_calls:
            final_text = resp.text
            break

        assistant_content = []
        if resp.text:
            assistant_content.append({"type": "text", "text": resp.text})
        for tc in resp.tool_calls:
            assistant_content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for tc in resp.tool_calls:
            ts = time.perf_counter()
            error = None
            try:
                result = dispatch_fn(tc.name, tc.arguments)
            except Exception as e:  # noqa: BLE001 — surfaced to the agent, not raised
                result = f"Error: {e}"
                error = str(e)
            te = time.perf_counter()
            tracer.record(
                kind="tool_call",
                name=tc.name,
                started_at=ts,
                ended_at=te,
                args=tc.arguments,
                result_repr=result,
                error=error,
            )
            tool_results.append({"type": "tool_result", "tool_use_id": tc.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    tracer.finish(final_answer=final_text, success=None)  # success graded by the eval harness
    return tracer.trace
