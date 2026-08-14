"""Ties pattern detection to a specific eval run and produces a ranked,
human-readable diagnosis report."""

from __future__ import annotations

from dataclasses import dataclass

from agentlens.diagnosis.patterns import Diagnosis, run_all_detectors
from agentlens.eval.harness import EvalResult


@dataclass
class DiagnosisReport:
    agent_name: str
    config_version: str
    diagnoses: list[Diagnosis]

    def summary_lines(self) -> list[str]:
        if not self.diagnoses:
            return [f"No failure patterns detected for {self.agent_name} ({self.config_version})."]
        lines = [f"Diagnosis for {self.agent_name} ({self.config_version}):"]
        for d in self.diagnoses:
            lines.append(
                f"  [{d.severity.upper()}] {d.pattern} — {len(d.affected_tasks)} task(s) affected"
            )
            lines.append(f"    {d.explanation}")
        return lines


def diagnose(eval_result: EvalResult) -> DiagnosisReport:
    diagnoses = run_all_detectors(eval_result)
    return DiagnosisReport(
        agent_name=eval_result.agent_name,
        config_version=eval_result.config_version,
        diagnoses=diagnoses,
    )
