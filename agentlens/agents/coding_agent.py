"""The coding agent — and its mock policy.

Like the research agent, the mock policy knows the right fix for each seeded
bug, but its *process* for getting there is deliberately inefficient unless
the system prompt contains the matching fix. The actual file edits and test
runs are real (see agentlens/tools/coding_tools.py) — only the "what should I
do next" decision is scripted.
"""

from __future__ import annotations

import re
import uuid

from agentlens.agents.base import AgentConfig, run_agent_task
from agentlens.eval.tasks import CodingTask
from agentlens.llm.client import LLMClient
from agentlens.tools.coding_tools import CodingToolkit
from agentlens.tracing.tracer import Trace

AGENT_NAME = "coding_agent"

PROMPT_FIXES = {
    "stop": (
        "As soon as the specific test you were asked to fix passes, stop immediately and "
        "report success; do not re-run tests or perform additional verification beyond "
        "what was requested."
    ),
    "reuse": (
        "Once you have read a file's contents, reuse them from context rather than "
        "re-reading the same file again unless you just modified it."
    ),
}

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant with read_file, write_file, list_dir, and run_tests "
    "tools, scoped to a small repository. Fix the bug described by the user, then "
    "verify your fix passes the relevant test."
)

FIXED_FUNCTIONS = {
    "add_task_mutable_default": (
        "add_task",
        'def add_task(tasks: list[dict], title: str, tags: list[str] | None = None) -> list[dict]:\n'
        '    if tags is None:\n'
        '        tags = ["untagged"]\n'
        '    tasks.append({"title": title, "tags": tags, "done": False})\n'
        '    return tasks\n',
    ),
    "complete_task_off_by_one": (
        "complete_task",
        'def complete_task(tasks: list[dict], title: str) -> list[dict]:\n'
        '    for t in tasks:\n'
        '        if t["title"] == title:\n'
        '            t["done"] = True\n'
        '    return tasks\n',
    ),
    "pending_tasks_inverted_filter": (
        "pending_tasks",
        'def pending_tasks(tasks: list[dict]) -> list[dict]:\n'
        '    return [t for t in tasks if not t["done"]]\n',
    ),
    "format_task_field_order": (
        "format_task",
        'def format_task(t: dict) -> str:\n'
        '    status = "done" if t["done"] else "pending"\n'
        '    return f"[{status}] {t[\'title\']}"\n',
    ),
}


def default_config(version: str = "v1") -> AgentConfig:
    return AgentConfig(version=version, system_prompt=BASE_SYSTEM_PROMPT, max_steps=10)


def _replace_function(source: str, func_name: str, new_func_source: str) -> str:
    pattern = re.compile(rf"^def {re.escape(func_name)}\(.*?(?=^def |\Z)", re.DOTALL | re.MULTILINE)
    new_source, n = pattern.subn(new_func_source.rstrip("\n") + "\n\n", source, count=1)
    if n == 0:
        raise ValueError(f"Could not find function {func_name!r} to replace")
    return new_source


def _collect_tool_results(messages: list[dict]) -> list[str]:
    results = []
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append(block["content"])
    return results


def make_coding_policy(task: CodingTask, system_prompt: str):
    stop_on = PROMPT_FIXES["stop"] in system_prompt
    reuse_on = PROMPT_FIXES["reuse"] in system_prompt

    context_rot = "context_rot" in task.inefficiency and not reuse_on
    retry_storm = "retry_storm" in task.inefficiency and not stop_on

    plan_types = ["read"] + (["read"] if context_rot else []) + ["write", "run_tests"]
    if retry_storm:
        plan_types += ["run_tests", "run_tests"]

    state = {"n": 0}

    def policy_fn(system: str, messages: list[dict], tools: list[dict]):
        idx = state["n"]
        if idx < len(plan_types):
            step_type = plan_types[idx]
            state["n"] += 1
            if step_type == "read":
                text = "Reading the file to see the current implementation." if idx == 0 else (
                    "Reading it again to double-check before I edit."
                )
                return text, [("read_file", {"path": "taskman/tasks.py"})]
            if step_type == "write":
                results = _collect_tool_results(messages)
                current_source = results[0]  # the first read_file result
                func_name, new_func_source = FIXED_FUNCTIONS[task.bug_key]
                fixed_source = _replace_function(current_source, func_name, new_func_source)
                return "Applying the fix.", [
                    ("write_file", {"path": "taskman/tasks.py", "content": fixed_source})
                ]
            if step_type == "run_tests":
                text = (
                    "Running the test to confirm the fix."
                    if idx == plan_types.index("run_tests")
                    else "Running it again just to be extra sure."
                )
                return text, [("run_tests", {"node_id": task.target_test_node_id})]

        return "Done — the fix is applied and the test passes.", []

    return policy_fn


def run_task(llm_client: LLMClient, config: AgentConfig, task: CodingTask) -> Trace:
    policy_fn = make_coding_policy(task, config.system_prompt)
    toolkit = CodingToolkit.fresh(run_id=f"{task.id}_{config.version}_{uuid.uuid4().hex[:8]}")

    def dispatch(name: str, args: dict) -> str:
        return toolkit.dispatch(name, args)

    return run_agent_task(
        llm_client=llm_client,
        agent_name=AGENT_NAME,
        config=config,
        tool_specs=toolkit.tool_specs(),
        dispatch_fn=dispatch,
        policy_fn=policy_fn,
        task_id=task.id,
        initial_prompt=task.prompt,
    )
