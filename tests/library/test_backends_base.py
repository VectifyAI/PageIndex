import asyncio

import pytest

from pageindex.backends.base import (CliBackend, CliBackendError, CliResult,
                                     strip_fences, looks_transient)


class EchoBackend(CliBackend):
    name = "echo"

    def build_command(self, workdir):
        return ["echo", self.model, workdir]

    def parse_output(self, result, workdir):
        if result.returncode != 0:
            raise CliBackendError("boom", retryable=looks_transient(result.stderr))
        return result.stdout


def make_run(script):
    """script: list of CliResult (or exceptions) returned per call."""
    calls = []

    async def run(cmd, stdin_text, timeout):
        calls.append((cmd, stdin_text, timeout))
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
    return run, calls


def test_strip_fences_removes_json_fence():
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('  plain  ') == 'plain'


def test_looks_transient():
    assert looks_transient("Error: rate limit exceeded")
    assert looks_transient("HTTP 529 overloaded")
    assert not looks_transient("invalid model name")


def test_acomplete_runs_command_with_prompt_on_stdin():
    run, calls = make_run([CliResult(0, "```\nhello\n```", "")])
    backend = EchoBackend("m1", run=run)
    assert asyncio.run(backend.acomplete("summarize this")) == "hello"
    cmd, stdin_text, timeout = calls[0]
    assert cmd[:2] == ["echo", "m1"]
    assert stdin_text == "summarize this"
    assert timeout == 600


def test_acomplete_retries_transient_then_succeeds(monkeypatch):
    sleeps = []

    async def no_sleep(seconds):
        sleeps.append(seconds)
    monkeypatch.setattr("pageindex.backends.base.asyncio.sleep", no_sleep)
    run, calls = make_run([CliResult(1, "", "rate limit"), CliResult(0, "ok", "")])
    backend = EchoBackend("m1", run=run, max_retries=3)
    assert asyncio.run(backend.acomplete("p")) == "ok"
    assert len(calls) == 2
    assert sleeps == [2]


def test_acomplete_gives_up_after_max_retries(monkeypatch):
    from pageindex.utils import LLMRetriesExhausted

    async def no_sleep(seconds):
        pass
    monkeypatch.setattr("pageindex.backends.base.asyncio.sleep", no_sleep)
    run, calls = make_run([CliResult(1, "", "rate limit")] * 3)
    backend = EchoBackend("m1", run=run, max_retries=3)
    with pytest.raises(LLMRetriesExhausted) as exc:
        asyncio.run(backend.acomplete("p"))
    assert exc.value.status_code is None
    assert len(calls) == 3


def test_acomplete_exhausted_ladder_is_unrecoverable(monkeypatch):
    """A sustained subscription failure typically carries no HTTP-style status
    code. utils._is_unrecoverable must still classify an exhausted CLI retry
    ladder as fatal, so summarize_tier aborts the run instead of burning
    through every remaining node's own retry ladder against the same
    exhausted rate limit."""
    from pageindex import utils

    async def no_sleep(seconds):
        pass
    monkeypatch.setattr("pageindex.backends.base.asyncio.sleep", no_sleep)
    run, calls = make_run([CliResult(1, "", "rate limit")] * 3)
    backend = EchoBackend("m1", run=run, max_retries=3)
    with pytest.raises(utils.LLMRetriesExhausted) as exc:
        asyncio.run(backend.acomplete("p"))
    assert utils._is_unrecoverable(exc.value) is True


def test_acomplete_does_not_retry_fatal_errors():
    run, calls = make_run([CliResult(1, "", "invalid model name")])
    backend = EchoBackend("m1", run=run, max_retries=3)
    with pytest.raises(CliBackendError):
        asyncio.run(backend.acomplete("p"))
    assert len(calls) == 1


def test_complete_is_sync_wrapper():
    run, _ = make_run([CliResult(0, "sync ok", "")])
    assert EchoBackend("m1", run=run).complete("p") == "sync ok"


def test_semaphore_limits_concurrency():
    active, peak = [0], [0]

    async def run(cmd, stdin_text, timeout):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.01)
        active[0] -= 1
        return CliResult(0, "x", "")

    backend = EchoBackend("m1", run=run, concurrency=2)

    async def many():
        await asyncio.gather(*(backend.acomplete("p") for _ in range(6)))
    asyncio.run(many())
    assert peak[0] == 2


def test_semaphore_survives_a_second_event_loop():
    run, _ = make_run([CliResult(0, "a", ""), CliResult(0, "b", "")])
    backend = EchoBackend("m1", run=run)
    assert asyncio.run(backend.acomplete("p")) == "a"
    assert asyncio.run(backend.acomplete("p")) == "b"


def test_resolve_and_register():
    from pageindex import backends
    assert backends.resolve("gpt-5.6-luna") is None
    assert backends.is_cli_model("claude-cli/sonnet")
    run, _ = make_run([CliResult(0, "r", "")])
    backends.register("echo-cli/x", EchoBackend("x", run=run))
    assert backends.resolve("echo-cli/x").complete("p") == "r"
