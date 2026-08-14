from agentlens.eval.checkers import check_excess_same_tool_calls, check_redundant_tool_calls
from agentlens.tracing.tracer import Tracer


def _trace_with_calls(calls: list[tuple[str, dict]]) -> "Trace":
    tracer = Tracer(agent_name="a", config_version="v1", task_id="t1")
    t = 0.0
    for name, args in calls:
        tracer.record(kind="tool_call", name=name, started_at=t, ended_at=t + 0.01, args=args)
        t += 0.01
    return tracer.finish(final_answer="x", success=True)


def test_redundant_tool_calls_flags_near_identical_search():
    trace = _trace_with_calls(
        [
            ("web_search", {"query": "population of france"}),
            ("web_search", {"query": "population of france - double-check"}),
        ]
    )
    findings = check_redundant_tool_calls(trace)
    assert len(findings) == 1
    assert findings[0].rule == "redundant_tool_calls"


def test_redundant_tool_calls_does_not_flag_distinct_queries():
    trace = _trace_with_calls(
        [
            ("web_search", {"query": "population of japan"}),
            ("web_search", {"query": "population of france"}),
        ]
    )
    findings = check_redundant_tool_calls(trace)
    assert findings == []


def test_excess_same_tool_calls_threshold():
    trace = _trace_with_calls([("run_tests", {"node_id": "x"})] * 3)
    findings = check_excess_same_tool_calls(trace, "run_tests", threshold=3)
    assert len(findings) == 1

    trace_two = _trace_with_calls([("run_tests", {"node_id": "x"})] * 2)
    assert check_excess_same_tool_calls(trace_two, "run_tests", threshold=3) == []
