"""Provider-agnostic LLM client interface.

Two implementations ship here:

- ``AnthropicClient`` — real tool-calling loop against the Claude API. This is
  the code path a real deployment/demo uses once ``ANTHROPIC_API_KEY`` is set.
- ``MockClient`` — a deterministic stand-in that requires no network access or
  API key. It does NOT fabricate metrics: token counts are estimated from the
  actual size of the prompts/messages passed through it (so a verbose system
  prompt genuinely costs more simulated tokens), and *what the agent decides
  to do* is delegated to a per-agent ``policy_fn`` that each target agent
  supplies. This is what lets the whole pipeline — tracer, eval harness,
  diagnosis engine, optimizer — run and produce a real, code-derived
  before/after delta today, with the exact same call sites working unchanged
  the moment a real API key is added. See README.md, "Why a mock provider?".
"""

from __future__ import annotations

import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    text: Optional[str]
    tool_calls: list[ToolCallRequest]
    usage: Usage
    latency_ms: float
    stop_reason: str = "end_turn"


# Rough, provider-agnostic token estimator (chars / 4). Good enough for
# relative before/after comparisons, which is all the harness needs — it
# never claims to match a specific tokenizer exactly.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        policy_fn: Optional[Callable] = None,
    ) -> LLMResponse:
        """Produce the next assistant turn given the conversation so far."""
        raise NotImplementedError


class MockClient(LLMClient):
    """Deterministic, offline stand-in for a real tool-calling LLM."""

    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)

    def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        policy_fn: Optional[Callable] = None,
    ) -> LLMResponse:
        if policy_fn is None:
            raise ValueError("MockClient requires a policy_fn (see agents/*.py)")

        start = time.perf_counter()
        text, tool_calls_raw = policy_fn(system=system, messages=messages, tools=tools)
        # Simulate the model "thinking" — proportional to how much context it
        # has to read, so a bloated prompt genuinely shows up as latency too.
        input_text = system + "".join(str(m) for m in messages)
        thinking_ms = 4 + estimate_tokens(input_text) * 0.02
        time.sleep(0)  # keep wall-clock cheap in CI; latency is still logged below
        elapsed_ms = thinking_ms + self._rng.uniform(2, 12)

        tool_calls = [
            ToolCallRequest(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(tool_calls_raw or [])
        ]

        output_text = text or ""
        output_repr = output_text + "".join(f"{tc.name}{tc.arguments}" for tc in tool_calls)

        usage = Usage(
            input_tokens=estimate_tokens(input_text),
            output_tokens=estimate_tokens(output_repr),
        )
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=elapsed_ms,
            stop_reason="tool_use" if tool_calls else "end_turn",
        )


class AnthropicClient(LLMClient):
    """Real Claude-backed tool-calling client. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: Optional[str] = None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for AGENTLENS_LLM_PROVIDER=anthropic. "
                "Run: pip install -r requirements.txt"
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or leave AGENTLENS_LLM_PROVIDER=mock to run without one."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def generate(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
        policy_fn: Optional[Callable] = None,
    ) -> LLMResponse:
        # policy_fn is ignored here — a real model decides its own actions.
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools or None,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, arguments=block.input))

        return LLMResponse(
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
            latency_ms=elapsed_ms,
            stop_reason=resp.stop_reason or "end_turn",
        )


def get_client() -> LLMClient:
    provider = os.environ.get("AGENTLENS_LLM_PROVIDER", "mock").lower()
    if provider == "anthropic":
        return AnthropicClient()
    return MockClient()
