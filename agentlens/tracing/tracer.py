"""Span-based tracing for agent runs.

Every LLM call and every tool call becomes a Span. A Trace is the ordered
list of spans for a single task run. This is deliberately shaped like what
production observability platforms (Braintrust, Langfuse, Helicone) capture,
so the vocabulary — span, trace, latency, token usage — carries directly into
an interview conversation about this project.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Span:
    kind: str  # "llm_call" | "tool_call"
    name: str
    started_at: float
    ended_at: float
    input_tokens: int = 0
    output_tokens: int = 0
    args: Optional[dict] = None
    result_repr: Optional[str] = None
    error: Optional[str] = None

    @property
    def latency_ms(self) -> float:
        return (self.ended_at - self.started_at) * 1000


@dataclass
class Trace:
    agent_name: str
    config_version: str
    task_id: str
    spans: list[Span] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: Optional[bool] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.spans)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.spans)

    @property
    def tool_call_count(self) -> int:
        return sum(1 for s in self.spans if s.kind == "tool_call")

    @property
    def llm_call_count(self) -> int:
        return sum(1 for s in self.spans if s.kind == "llm_call")

    def tool_calls(self) -> list[Span]:
        return [s for s in self.spans if s.kind == "tool_call"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_input_tokens"] = self.total_input_tokens
        d["total_output_tokens"] = self.total_output_tokens
        d["total_tokens"] = self.total_tokens
        d["total_latency_ms"] = self.total_latency_ms
        d["tool_call_count"] = self.tool_call_count
        return d


class Tracer:
    """Collects spans for the current task run and persists traces to JSONL."""

    def __init__(self, agent_name: str, config_version: str, task_id: str):
        self.trace = Trace(agent_name=agent_name, config_version=config_version, task_id=task_id)

    def record(
        self,
        *,
        kind: str,
        name: str,
        started_at: float,
        ended_at: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        args: Optional[dict] = None,
        result_repr: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Span:
        span = Span(
            kind=kind,
            name=name,
            started_at=started_at,
            ended_at=ended_at,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            args=args,
            result_repr=(result_repr[:400] if result_repr else result_repr),
            error=error,
        )
        self.trace.spans.append(span)
        return span

    def finish(self, *, final_answer: Optional[str], success: Optional[bool]) -> Trace:
        self.trace.final_answer = final_answer
        self.trace.success = success
        self.trace.ended_at = time.time()
        return self.trace


def write_traces(traces: list[Trace], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), default=str) + "\n")


def read_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
