"""LLM backend for Kirill's Task Engine that runs the Claude Code CLI on the owner's Claude subscription
(no metered API key). Same signature as task_engine.compiler.claude_backend: (system, messages) -> TaskSpecDraft.

Auth inside the container: CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`, long-lived, subscription-bound),
passed through ~/.zalu_env. Structured output uses `claude -p --json-schema`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import List

CLAUDE_BIN = os.environ.get("ZALU_CLAUDE_BIN") or shutil.which("claude") or "/root/.local/bin/claude"
TIMEOUT_S = int(os.environ.get("ZALU_CLAUDE_TIMEOUT", "180"))


class ClaudeCLIError(RuntimeError):
    pass


def available() -> bool:
    return bool(os.path.exists(CLAUDE_BIN) and (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
                                                or os.path.exists(os.path.expanduser("~/.claude/.credentials.json"))))


def _render(messages: List[dict]) -> str:
    """Flatten a short chat (first request + repair rounds) into one prompt for `claude -p`."""
    if len(messages) == 1:
        return str(messages[0]["content"])
    parts = []
    for m in messages:
        who = "REQUEST" if m["role"] == "user" else "YOUR PREVIOUS ANSWER"
        parts.append(f"### {who}\n{m['content']}")
    parts.append("### Return the corrected answer only.")
    return "\n\n".join(parts)


def claude_cli_backend(model: str | None = None):
    from backend.task_engine.schema import TaskSpecDraft

    schema = json.dumps(TaskSpecDraft.model_json_schema())

    def call(system: str, messages: List[dict]):
        cmd = [CLAUDE_BIN, "-p", "--output-format", "json", "--json-schema", schema, "--system-prompt", system]
        if model:
            cmd += ["--model", model]
        try:
            p = subprocess.run(cmd + [_render(messages)], capture_output=True, text=True, timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise ClaudeCLIError(f"claude CLI timed out after {TIMEOUT_S}s")
        if p.returncode != 0:
            raise ClaudeCLIError(f"claude CLI exit {p.returncode}: {(p.stderr or p.stdout)[-500:]}")
        try:
            out = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise ClaudeCLIError(f"claude CLI returned non-JSON: {p.stdout[:300]}")
        if out.get("is_error"):
            raise ClaudeCLIError(f"claude CLI error: {str(out.get('result'))[:300]}")
        data = out.get("structured_output")
        if data is None:
            raise ClaudeCLIError("claude CLI returned no structured_output")
        return TaskSpecDraft.model_validate(data)

    call.name = f"claude-cli:{model or 'default'}"  # type: ignore[attr-defined]
    return call
