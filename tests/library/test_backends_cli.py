import asyncio
import json
import os

import pytest

from pageindex.backends.base import CliBackendError, CliResult
from pageindex.backends.claude_cli import ClaudeCliBackend
from pageindex.backends.codex_cli import CodexCliBackend


def scripted(results):
    calls = []

    async def run(cmd, stdin_text, timeout):
        calls.append((cmd, stdin_text))
        return results.pop(0)
    return run, calls


def claude_json(result, is_error=False):
    return json.dumps({"type": "result", "subtype": "success",
                       "is_error": is_error, "result": result})


def test_claude_command_line_minimizes_overhead():
    run, calls = scripted([CliResult(0, claude_json("ok"), "")])
    backend = ClaudeCliBackend("sonnet", run=run, system_prompt="SYS")
    assert asyncio.run(backend.acomplete("hello")) == "ok"
    cmd, stdin_text = calls[0]
    assert cmd[0] == "claude"
    assert "-p" in cmd
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--output-format") + 1] == "json"
    assert cmd[cmd.index("--system-prompt") + 1] == "SYS"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert "--no-session-persistence" in cmd
    assert "--bare" not in cmd
    assert stdin_text == "hello"


def test_claude_strips_json_fence():
    run, _ = scripted([CliResult(0, claude_json('```json\n{"summary": "s"}\n```'), "")])
    assert asyncio.run(ClaudeCliBackend("sonnet", run=run).acomplete("p")) == '{"summary": "s"}'


def test_claude_not_logged_in_is_fatal_401():
    run, calls = scripted([CliResult(0, claude_json("Not logged in · Please run /login", True), "")])
    with pytest.raises(CliBackendError) as exc:
        asyncio.run(ClaudeCliBackend("sonnet", run=run, max_retries=3).acomplete("p"))
    assert exc.value.status_code == 401
    assert len(calls) == 1


def test_claude_non_json_output_is_retryable(monkeypatch):
    async def no_sleep(s):
        pass
    monkeypatch.setattr("pageindex.backends.base.asyncio.sleep", no_sleep)
    run, calls = scripted([CliResult(1, "", "connection reset"),
                           CliResult(0, claude_json("fine"), "")])
    assert asyncio.run(ClaudeCliBackend("sonnet", run=run).acomplete("p")) == "fine"
    assert len(calls) == 2


def test_codex_command_line_and_last_message_file():
    captured = {}

    async def run(cmd, stdin_text, timeout):
        captured["cmd"] = cmd
        captured["stdin"] = stdin_text
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w", encoding="utf-8") as f:
            f.write("```\nfrom codex\n```")
        return CliResult(0, "tokens used\n10", "")

    backend = CodexCliBackend("gpt-5.6-luna", run=run, system_prompt="SYS")
    assert asyncio.run(backend.acomplete("hello")) == "from codex"
    cmd = captured["cmd"]
    assert cmd[:2] == ["codex", "exec"]
    for flag in ("--ephemeral", "--skip-git-repo-check"):
        assert flag in cmd
    assert cmd[cmd.index("-s") + 1] == "read-only"
    assert cmd[cmd.index("-m") + 1] == "gpt-5.6-luna"
    assert cmd[cmd.index("-c") + 1] == "model_reasoning_effort=low"
    assert cmd[-1] == "-"
    workdir = cmd[cmd.index("-C") + 1]
    assert cmd[cmd.index("-o") + 1] == os.path.join(workdir, "last.txt")
    assert captured["stdin"].startswith("SYS\n\n")
    assert captured["stdin"].endswith("hello")


def test_codex_empty_last_message_raises():
    async def run(cmd, stdin_text, timeout):
        return CliResult(1, "", "invalid model")
    with pytest.raises(CliBackendError) as exc:
        asyncio.run(CodexCliBackend("m", run=run).acomplete("p"))
    assert exc.value.retryable is False
