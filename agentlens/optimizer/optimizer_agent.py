"""The optimizer: reads a DiagnosisReport and produces a new, patched
AgentConfig — this is the part of AgentLens that is itself agentic. It
doesn't just flag problems, it acts on them and hands back something you can
re-evaluate.

FIX_LIBRARY maps each diagnosis's recommended_fix_key to the exact prompt
instruction that resolves it (the same PROMPT_FIXES strings each target
agent's mock policy checks for — see agents/research_agent.py and
coding_agent.py). In mock mode the mapping is a lookup; in AGENTLENS_LLM_PROVIDER
=anthropic mode, ``propose_patch`` instead asks Claude to rewrite the prompt
given the diagnosis, and the library becomes a fallback/reference rather than
the source of truth. Both paths return the same PatchResult shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agentlens.agents.base import AgentConfig
from agentlens.agents.coding_agent import PROMPT_FIXES as CODING_FIXES
from agentlens.agents.research_agent import PROMPT_FIXES as RESEARCH_FIXES
from agentlens.diagnosis.diagnoser import DiagnosisReport

FIX_LIBRARY = {
    "dedupe_tool_calls": RESEARCH_FIXES["dedupe"],
    "concise_reasoning": RESEARCH_FIXES["concise"],
    "stop_on_success": CODING_FIXES["stop"],
    "reuse_context": CODING_FIXES["reuse"],
}


@dataclass
class PatchResult:
    new_config: AgentConfig
    applied_fixes: list[str]  # recommended_fix_key values that were applied
    diff: str  # human-readable summary of what changed


def propose_patch(base_config: AgentConfig, report: DiagnosisReport, new_version: str) -> PatchResult:
    provider = os.environ.get("AGENTLENS_LLM_PROVIDER", "mock").lower()
    if provider == "anthropic":
        return _propose_patch_llm(base_config, report, new_version)
    return _propose_patch_rule_based(base_config, report, new_version)


def _propose_patch_rule_based(
    base_config: AgentConfig, report: DiagnosisReport, new_version: str
) -> PatchResult:
    additions = []
    applied = []
    for d in report.diagnoses:
        fix_text = FIX_LIBRARY.get(d.recommended_fix_key)
        if fix_text and fix_text not in base_config.system_prompt:
            additions.append(fix_text)
            applied.append(d.recommended_fix_key)

    if not additions:
        return PatchResult(new_config=base_config, applied_fixes=[], diff="No changes — no open diagnoses.")

    new_prompt = base_config.system_prompt + "\n" + "\n".join(additions)
    new_config = AgentConfig(version=new_version, system_prompt=new_prompt, max_steps=base_config.max_steps)
    diff_lines = [f"+ {line}" for line in additions]
    diff = (
        f"--- {base_config.version} system_prompt\n+++ {new_version} system_prompt\n"
        + "\n".join(diff_lines)
    )
    return PatchResult(new_config=new_config, applied_fixes=applied, diff=diff)


def _propose_patch_llm(base_config: AgentConfig, report: DiagnosisReport, new_version: str) -> PatchResult:
    """Real path: ask Claude to rewrite the prompt given the diagnosis. Falls
    back to the rule-based library for any pattern it doesn't address, so a
    real run never regresses below what the deterministic path guarantees."""
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    diagnosis_text = "\n".join(report.summary_lines())
    prompt = (
        "You are optimizing a system prompt for an LLM agent based on a diagnosis of "
        "its inefficiencies. Rewrite the system prompt to fix the issues below while "
        "preserving its original intent. Return ONLY the new system prompt text.\n\n"
        f"Current system prompt:\n{base_config.system_prompt}\n\n"
        f"Diagnosis:\n{diagnosis_text}"
    )
    resp = client.messages.create(
        model=model, max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )
    new_prompt = "".join(b.text for b in resp.content if b.type == "text").strip()
    new_config = AgentConfig(version=new_version, system_prompt=new_prompt, max_steps=base_config.max_steps)
    applied = [d.recommended_fix_key for d in report.diagnoses]
    diff = f"--- {base_config.version}\n+++ {new_version}\n(LLM-rewritten prompt; see new_config.system_prompt)"
    return PatchResult(new_config=new_config, applied_fixes=applied, diff=diff)
