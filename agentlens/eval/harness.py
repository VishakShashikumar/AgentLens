"""Runs a target agent's full golden-task suite against one config version and
aggregates traces, rule findings, and judge verdicts into a single EvalResult.
This is the object both the diagnosis engine and the report card consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentlens.agents import coding_agent, research_agent
from agentlens.agents.base import AgentConfig
from agentlens.eval.checkers import RuleFinding, run_all_checks
from agentlens.eval.judge import JudgeVerdict, judge_trace
from agentlens.eval.tasks import CODING_TASKS, RESEARCH_TASKS
from agentlens.llm.client import LLMClient
from agentlens.tracing.tracer import Trace

CODING_IDEAL_CALL_COUNT = 3  # read_file, write_file, run_tests


@dataclass
class EvalResult:
    agent_name: str
    config_version: str
    traces: list[Trace] = field(default_factory=list)
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    rule_findings: list[RuleFinding] = field(default_factory=list)
    judge_tokens: int = 0

    @property
    def n_tasks(self) -> int:
        return len(self.traces)

    @property
    def success_rate(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.task_success) / len(self.verdicts)

    @property
    def avg_quality_score(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(v.quality_score for v in self.verdicts) / len(self.verdicts)

    @property
    def total_agent_tokens(self) -> int:
        return sum(t.total_tokens for t in self.traces)

    @property
    def avg_agent_tokens(self) -> float:
        return self.total_agent_tokens / self.n_tasks if self.n_tasks else 0.0

    @property
    def total_tool_calls(self) -> int:
        return sum(t.tool_call_count for t in self.traces)

    @property
    def avg_latency_ms(self) -> float:
        if not self.traces:
            return 0.0
        return sum(t.total_latency_ms for t in self.traces) / len(self.traces)

    def verdict_for(self, task_id: str) -> JudgeVerdict | None:
        return next((v for v in self.verdicts if v.task_id == task_id), None)

    def trace_for(self, task_id: str) -> Trace | None:
        return next((t for t in self.traces if t.task_id == task_id), None)


def run_research_eval(llm_client: LLMClient, config: AgentConfig) -> EvalResult:
    traces, verdicts = [], []
    judge_tokens = 0
    for task in RESEARCH_TASKS:
        trace = research_agent.run_task(llm_client, config, task)
        traces.append(trace)
        verdict, usage = judge_trace(
            llm_client, trace, task, "research", ideal_call_count=len(task.ideal_plan)
        )
        verdicts.append(verdict)
        judge_tokens += usage.total_tokens
    rule_findings = run_all_checks(traces)
    return EvalResult(
        agent_name="research_agent",
        config_version=config.version,
        traces=traces,
        verdicts=verdicts,
        rule_findings=rule_findings,
        judge_tokens=judge_tokens,
    )


def run_coding_eval(llm_client: LLMClient, config: AgentConfig) -> EvalResult:
    traces, verdicts = [], []
    judge_tokens = 0
    for task in CODING_TASKS:
        trace = coding_agent.run_task(llm_client, config, task)
        traces.append(trace)
        verdict, usage = judge_trace(
            llm_client, trace, task, "coding", ideal_call_count=CODING_IDEAL_CALL_COUNT
        )
        verdicts.append(verdict)
        judge_tokens += usage.total_tokens
    rule_findings = run_all_checks(traces)
    return EvalResult(
        agent_name="coding_agent",
        config_version=config.version,
        traces=traces,
        verdicts=verdicts,
        rule_findings=rule_findings,
        judge_tokens=judge_tokens,
    )
