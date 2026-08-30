"""`claude -p` (Claude Code headless) as a completion backend.

Runs on the Claude subscription login. The flag set matters: the default
Claude Code system prompt, tools, settings and MCP servers add ~29k tokens
per call; with --system-prompt/--tools ""/--setting-sources ""/empty MCP
config the overhead is ~250 tokens. --bare is NOT used: it skips the stored
login and fails with "Not logged in"."""
from __future__ import annotations

import json

from .base import CliBackend, CliBackendError, CliResult, looks_transient

EMPTY_MCP_CONFIG = '{"mcpServers":{}}'


class ClaudeCliBackend(CliBackend):
    name = "claude-cli"
    executable = "claude"

    def build_command(self, workdir: str) -> list[str]:
        return [
            self.executable, "-p",
            "--output-format", "json",
            "--model", self.model,
            "--no-session-persistence",
            "--system-prompt", self.system_prompt,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config", "--mcp-config", EMPTY_MCP_CONFIG,
        ]

    def parse_output(self, result: CliResult, workdir: str) -> str:
        stdout = result.stdout.strip()
        if not stdout:
            raise CliBackendError(
                f"claude exited {result.returncode}: {result.stderr.strip()[-500:]}",
                retryable=looks_transient(result.stderr) or result.returncode != 0)
        try:
            data = json.loads(stdout)
        except ValueError:
            raise CliBackendError(
                f"claude returned non-JSON output: {stdout[:200]!r}")
        text = data.get("result") or ""
        if data.get("is_error"):
            if "not logged in" in text.lower():
                raise CliBackendError(
                    f"claude: {text} — run `claude` once and /login", status_code=401,
                    retryable=False)
            raise CliBackendError(f"claude error: {text[:300]}",
                                  retryable=looks_transient(text))
        if not text.strip():
            raise CliBackendError("claude returned an empty result")
        return text
