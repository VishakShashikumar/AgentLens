from agentlens.agents import research_agent
from agentlens.diagnosis.diagnoser import diagnose
from agentlens.eval import harness
from agentlens.llm.client import MockClient


def test_v1_research_agent_shows_prompt_bloat_and_ping_pong():
    client = MockClient()
    result = harness.run_research_eval(client, research_agent.default_config("v1"))
    report = diagnose(result)
    patterns = {d.pattern for d in report.diagnoses}
    assert "Prompt Bloat" in patterns
    assert "Tool-Call Ping-Pong" in patterns


def test_v2_research_agent_with_fixes_is_clean():
    client = MockClient()
    config = research_agent.default_config("v2")
    config.system_prompt += (
        "\n" + research_agent.PROMPT_FIXES["dedupe"] + "\n" + research_agent.PROMPT_FIXES["concise"]
    )
    result = harness.run_research_eval(client, config)
    report = diagnose(result)
    assert report.diagnoses == []
