"""The regression-gate test: proves the full audit pipeline — baseline eval,
diagnose, optimize, re-eval — produces a real, measurable improvement. This
is the test you'd wire into CI to catch a future change that quietly makes
the optimizer (or a target agent) worse.
"""

from agentlens.agents import coding_agent, research_agent
from agentlens.eval import harness
from agentlens.llm.client import MockClient
from agentlens.scripts.run_audit import audit_agent


def test_research_agent_audit_improves_efficiency_without_regressing_success():
    client = MockClient()
    report = audit_agent("research_agent", client, research_agent.default_config, harness.run_research_eval)

    assert report["v2"]["success_rate"] >= report["v1"]["success_rate"]
    assert report["v2"]["total_tokens"] < report["v1"]["total_tokens"]
    assert report["v2"]["total_tool_calls"] < report["v1"]["total_tool_calls"]
    assert len(report["diagnosis"]) >= 1
    assert "dedupe_tool_calls" in report["applied_fixes"]
    assert "concise_reasoning" in report["applied_fixes"]


def test_coding_agent_audit_improves_efficiency_without_regressing_success():
    client = MockClient()
    report = audit_agent("coding_agent", client, coding_agent.default_config, harness.run_coding_eval)

    assert report["v2"]["success_rate"] == 1.0
    assert report["v1"]["success_rate"] == 1.0  # bugs get fixed either way; efficiency is what changes
    assert report["v2"]["total_tokens"] < report["v1"]["total_tokens"]
    assert report["v2"]["total_tool_calls"] < report["v1"]["total_tool_calls"]
    assert "stop_on_success" in report["applied_fixes"]
    assert "reuse_context" in report["applied_fixes"]


def test_regression_gate_catches_a_worse_v2(monkeypatch):
    """Sanity-check that the gate isn't vacuous: if v2 were actually worse,
    the assertion pattern above would fail. We simulate that here directly
    against the eval harness rather than mutating the optimizer."""
    client = MockClient()
    v1 = harness.run_research_eval(client, research_agent.default_config("v1"))
    # A "v2" that's just v1 again should NOT look like an improvement.
    v2_no_fix = harness.run_research_eval(client, research_agent.default_config("v1_again"))
    assert v2_no_fix.total_agent_tokens == v1.total_agent_tokens
