"""A tiny task-list library — intentionally seeded with four bugs.

Each bug is targeted by exactly one golden task for the coding agent
(see agentlens/eval/tasks.py). The agent reads this file for real, writes a
real fix, and the fix is verified by running the real pytest suite in
sandbox_repo/tests/test_tasks.py — none of this is simulated.
"""

from __future__ import annotations


def add_task(tasks: list[dict], title: str, tags: list[str] = []) -> list[dict]:
    # BUG: classic mutable default argument. `tags` defaults to a single list
    # object created once at import time; every call that omits `tags`
    # mutates and shares that *same* object, so tags leak across tasks.
    tags.append("untagged")
    tasks.append({"title": title, "tags": tags, "done": False})
    return tasks


def complete_task(tasks: list[dict], title: str) -> list[dict]:
    # BUG: off-by-one — iterates over tasks[:-1], so completing the *last*
    # task in the list silently does nothing.
    for t in tasks[:-1]:
        if t["title"] == title:
            t["done"] = True
    return tasks


def pending_tasks(tasks: list[dict]) -> list[dict]:
    # BUG: inverted filter — returns completed tasks instead of pending ones.
    return [t for t in tasks if t["done"]]


def format_task(t: dict) -> str:
    # BUG: wrong field order — status should come first, e.g. "[pending] Buy milk".
    status = "done" if t["done"] else "pending"
    return f"{t['title']} [{status}]"
