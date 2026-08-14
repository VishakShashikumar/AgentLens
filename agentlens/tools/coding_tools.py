"""Real filesystem + pytest tools for the coding agent.

Unlike the research tools, these are NOT simulated: read_file/write_file
touch real files in a per-run temp copy of sandbox_repo/, and run_tests()
shells out to a real pytest subprocess. Only the *decision* of what to read,
write, and run is driven by a policy (mock or real LLM) — the actions
themselves, and their pass/fail outcomes and latency, are genuine.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

SANDBOX_SOURCE = Path(__file__).resolve().parent.parent.parent / "sandbox_repo"


class CodingToolkit:
    def __init__(self, run_root: Path):
        self.root = run_root
        self._read_cache: dict[str, str] = {}
        self.real_test_runs: int = 0

    @classmethod
    def fresh(cls, run_id: str) -> "CodingToolkit":
        run_root = Path("/tmp/agentlens_runs") / run_id
        if run_root.exists():
            shutil.rmtree(run_root)
        shutil.copytree(SANDBOX_SOURCE, run_root)
        return cls(run_root)

    def read_file(self, path: str) -> str:
        full = self.root / path
        content = full.read_text()
        self._read_cache[path] = content
        return content

    def write_file(self, path: str, content: str) -> str:
        full = self.root / path
        full.write_text(content)
        self._read_cache[path] = content
        return f"Wrote {len(content)} chars to {path}."

    def list_dir(self, path: str = ".") -> str:
        full = self.root / path
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in full.iterdir())
        return "\n".join(entries)

    def run_tests(self, node_id: str = "") -> str:
        self.real_test_runs += 1
        pytest_bin = shutil.which("pytest") or "pytest"
        args = [pytest_bin, "-q"]
        if node_id:
            args.append(node_id)
        proc = subprocess.run(
            args, cwd=self.root, capture_output=True, text=True, timeout=30
        )
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        status = "PASSED" if proc.returncode == 0 else "FAILED"
        return f"{status}\n{tail}"

    def tool_specs(self) -> list[dict]:
        return [
            {
                "name": "read_file",
                "description": "Read a file's contents, relative to the repo root.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Overwrite a file with new contents, relative to the repo root.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "list_dir",
                "description": "List files in a directory, relative to the repo root.",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
            {
                "name": "run_tests",
                "description": "Run the pytest suite (optionally a specific node id) and report pass/fail.",
                "input_schema": {
                    "type": "object",
                    "properties": {"node_id": {"type": "string"}},
                },
            },
        ]

    def dispatch(self, name: str, args: dict) -> str:
        fn = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_dir": self.list_dir,
            "run_tests": self.run_tests,
        }[name]
        return fn(**args)
