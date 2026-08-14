"""Golden task definitions for both target agents.

Each task is a fixed, hand-checkable unit of work: a research question with
required substrings in the answer, or a coding bug with a specific pytest
node id that must pass. ``inefficiency`` flags which seeded bad habits a
mock run exhibits *when the current agent config is missing the fix for it*
— see agentlens/agents/research_agent.py and coding_agent.py for how those
flags translate into behavior, and agentlens/optimizer/optimizer_agent.py
for how the fixes get written back into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ResearchTask:
    id: str
    prompt: str
    ideal_plan: list[tuple[str, dict]]
    compose_answer: Callable[[list[str]], str]
    answer_contains: list[str]
    inefficiency: set[str] = field(default_factory=set)


@dataclass
class CodingTask:
    id: str
    prompt: str
    bug_key: str
    target_test_node_id: str
    inefficiency: set[str] = field(default_factory=set)


RESEARCH_TASKS: list[ResearchTask] = [
    ResearchTask(
        id="r1_france_population",
        prompt="What is the population of France? Answer in a single sentence.",
        ideal_plan=[("web_search", {"query": "population of france"})],
        compose_answer=lambda r: f"France's population is approximately 68 million.",
        answer_contains=["68 million"],
        inefficiency={"duplicate_search", "verbose_reasoning"},
    ),
    ResearchTask(
        id="r2_eu_states_per_capita",
        prompt=(
            "France's population is about 68 million and the EU has 27 member states. "
            "About how many million people is that per member state? Look up the EU "
            "member state count to confirm, then divide."
        ),
        ideal_plan=[
            ("web_search", {"query": "eu member states count"}),
            ("calculator", {"expression": "68 / 27"}),
        ],
        compose_answer=lambda r: f"That's about {r[-1][:5]} million people per EU member state.",
        answer_contains=["2.51"],
        inefficiency={"duplicate_search", "verbose_reasoning"},
    ),
    ResearchTask(
        id="r3_nitrogen_boiling_f",
        prompt="What is the boiling point of nitrogen, converted to Fahrenheit?",
        ideal_plan=[
            ("web_search", {"query": "boiling point of nitrogen"}),
            ("calculator", {"expression": "-195.8 * 9 / 5 + 32"}),
        ],
        compose_answer=lambda r: f"Nitrogen boils at about {r[-1][:7]} degrees Fahrenheit.",
        answer_contains=["-320"],
        inefficiency={"duplicate_search", "verbose_reasoning"},
    ),
    ResearchTask(
        id="r4_light_speed_note",
        prompt="Look up the speed of light and save it as a note for later reference.",
        ideal_plan=[
            ("web_search", {"query": "speed of light"}),
            ("save_note", {"text": "Speed of light: 299,792,458 m/s"}),
        ],
        compose_answer=lambda r: "Saved: the speed of light is 299,792,458 meters per second.",
        answer_contains=["299,792,458"],
        inefficiency={"verbose_reasoning"},
    ),
    ResearchTask(
        id="r5_everest_height",
        prompt="How tall is the tallest mountain above sea level?",
        ideal_plan=[("web_search", {"query": "tallest mountain"})],
        compose_answer=lambda r: "The tallest mountain above sea level is Mount Everest, at 8,849 meters.",
        answer_contains=["8,849", "everest"],
        inefficiency={"duplicate_search", "verbose_reasoning"},
    ),
    ResearchTask(
        id="r6_japan_vs_france_pop",
        prompt=(
            "Look up the populations of Japan and France, then calculate how many more "
            "million people Japan has than France."
        ),
        ideal_plan=[
            ("web_search", {"query": "population of japan"}),
            ("web_search", {"query": "population of france"}),
            ("calculator", {"expression": "124 - 68"}),
        ],
        compose_answer=lambda r: f"Japan has about {r[-1]} million more people than France.",
        answer_contains=["56"],
        inefficiency={"verbose_reasoning"},
    ),
    ResearchTask(
        id="r7_us_states_count",
        prompt="How many states does the United States have?",
        ideal_plan=[("web_search", {"query": "number of us states"})],
        compose_answer=lambda r: "The United States has 50 states.",
        answer_contains=["50"],
        inefficiency={"duplicate_search", "verbose_reasoning"},
    ),
    ResearchTask(
        id="r8_mars_distance_note",
        prompt="Look up Mars's average distance from the Sun and save it as a note.",
        ideal_plan=[
            ("web_search", {"query": "mars distance from sun"}),
            ("save_note", {"text": "Mars: 227.9 million km from the Sun"}),
        ],
        compose_answer=lambda r: "Saved: Mars orbits at an average of 227.9 million km from the Sun.",
        answer_contains=["227.9"],
        inefficiency={"verbose_reasoning"},
    ),
    ResearchTask(
        id="r9_python_creator",
        prompt="Who created Python, and in what year was it first released?",
        ideal_plan=[("web_search", {"query": "python creator"})],
        compose_answer=lambda r: "Python was created by Guido van Rossum and first released in 1991.",
        answer_contains=["guido van rossum", "1991"],
        inefficiency={"duplicate_search", "verbose_reasoning"},
    ),
    ResearchTask(
        id="r10_water_vs_nitrogen_gap",
        prompt=(
            "Water boils at 100C and nitrogen boils at -195.8C. What is the gap in "
            "degrees Celsius between the two boiling points?"
        ),
        ideal_plan=[
            ("web_search", {"query": "boiling point of water"}),
            ("web_search", {"query": "boiling point of nitrogen"}),
            ("calculator", {"expression": "100 - -195.8"}),
        ],
        compose_answer=lambda r: f"The gap is {r[-1]} degrees Celsius.",
        answer_contains=["295.8"],
        inefficiency={"verbose_reasoning"},
    ),
]

CODING_TASKS: list[CodingTask] = [
    CodingTask(
        id="c1_add_task_mutable_default",
        prompt=(
            "taskman/tasks.py: add_task() leaks tags between tasks that don't pass an "
            "explicit tags list. Fix it so each task gets its own independent tags list."
        ),
        bug_key="add_task_mutable_default",
        target_test_node_id="tests/test_tasks.py::test_add_task_does_not_leak_tags_across_calls",
        inefficiency={"retry_storm", "context_rot"},
    ),
    CodingTask(
        id="c2_complete_task_off_by_one",
        prompt=(
            "taskman/tasks.py: complete_task() fails to mark the last task in the list "
            "as done. Fix the loop so it checks every task."
        ),
        bug_key="complete_task_off_by_one",
        target_test_node_id="tests/test_tasks.py::test_complete_task_marks_last_task_done",
        inefficiency={"retry_storm"},
    ),
    CodingTask(
        id="c3_pending_tasks_inverted_filter",
        prompt=(
            "taskman/tasks.py: pending_tasks() returns completed tasks instead of "
            "pending ones. Fix the filter condition."
        ),
        bug_key="pending_tasks_inverted_filter",
        target_test_node_id="tests/test_tasks.py::test_pending_tasks_excludes_done",
        inefficiency={"retry_storm", "context_rot"},
    ),
    CodingTask(
        id="c4_format_task_field_order",
        prompt=(
            "taskman/tasks.py: format_task() prints the title before the status. Fix it "
            "to print the status first, e.g. '[pending] Buy milk'."
        ),
        bug_key="format_task_field_order",
        target_test_node_id="tests/test_tasks.py::test_format_task_output_format",
        inefficiency={"retry_storm"},
    ),
]
