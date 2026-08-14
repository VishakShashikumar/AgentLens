"""Tools for the research agent.

``web_search`` is a deterministic local knowledge base standing in for a real
search API (Tavily, Serper, Bing) — swap ``_KNOWLEDGE_BASE`` lookups for a
real HTTP call once you're running against a live model; nothing else in the
agent changes, since the tool's contract (query in, snippet out) is the same
either way.
"""

from __future__ import annotations

import ast
import operator

_KNOWLEDGE_BASE = {
    "population of france": "France's population is approximately 68 million (2024 estimate).",
    "eu member states count": "The European Union has 27 member states.",
    "boiling point of nitrogen": "Nitrogen boils at -195.8 degrees Celsius at standard pressure.",
    "speed of light": "The speed of light in a vacuum is 299,792,458 meters per second.",
    "tallest mountain": "Mount Everest is the tallest mountain above sea level, at 8,849 meters.",
    "population of japan": "Japan's population is approximately 124 million (2024 estimate).",
    "number of us states": "The United States has 50 states.",
    "mars distance from sun": "Mars orbits at an average distance of 227.9 million kilometers from the Sun.",
    "python creator": "Python was created by Guido van Rossum, first released in 1991.",
    "boiling point of water": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
}


def web_search(query: str) -> str:
    q = query.lower().strip()
    for key, fact in _KNOWLEDGE_BASE.items():
        if key in q or all(word in q for word in key.split()[:2]):
            return fact
    return f"No results found for '{query}'."


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


_notes: list[str] = []


def save_note(text: str) -> str:
    _notes.append(text)
    return "Note saved."


RESEARCH_TOOL_SPECS = [
    {
        "name": "web_search",
        "description": "Search the web for a fact. Returns a short snippet.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a numeric arithmetic expression, e.g. '68 / 27'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "save_note",
        "description": "Save an intermediate finding to scratch notes.",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]

RESEARCH_TOOL_IMPL = {
    "web_search": web_search,
    "calculator": calculator,
    "save_note": save_note,
}
