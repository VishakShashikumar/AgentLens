from agentlens.tracing.tracer import Tracer


def test_trace_aggregates_tokens_and_latency():
    tracer = Tracer(agent_name="test_agent", config_version="v1", task_id="t1")
    tracer.record(kind="llm_call", name="turn", started_at=0.0, ended_at=0.01, input_tokens=100, output_tokens=20)
    tracer.record(kind="tool_call", name="web_search", started_at=0.01, ended_at=0.02, args={"query": "x"})
    tracer.record(kind="llm_call", name="turn", started_at=0.02, ended_at=0.03, input_tokens=150, output_tokens=5)
    trace = tracer.finish(final_answer="done", success=True)

    assert trace.total_input_tokens == 250
    assert trace.total_output_tokens == 25
    assert trace.total_tokens == 275
    assert trace.tool_call_count == 1
    assert trace.llm_call_count == 2
    assert trace.total_latency_ms > 0


def test_trace_to_dict_is_json_safe():
    import json

    tracer = Tracer(agent_name="a", config_version="v1", task_id="t1")
    tracer.record(kind="tool_call", name="calculator", started_at=0.0, ended_at=0.001, args={"expression": "1+1"}, result_repr="2")
    trace = tracer.finish(final_answer="2", success=True)
    json.dumps(trace.to_dict())  # must not raise
