"""Subprocess-backed model backends: one CLI process per completion.

A backend turns a prompt into a command line, feeds the prompt on stdin,
and parses the process output. Concurrency is a per-event-loop semaphore
(summaries run under asyncio.run() several times per process, and an
asyncio.Semaphore bound to a finished loop cannot be reused)."""
from __future__ import annotations

import asyncio
import re
import tempfile
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful reader summarizing sections of books. Follow the "
    "user's output format exactly and reply with the requested content only."
)

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.S)
_TRANSIENT_MARKERS = ("rate limit", "rate_limit", "429", "529", "overloaded",
                      "usage limit", "too many requests", "timed out",
                      "timeout", "connection", "temporarily")


@dataclass
class CliResult:
    returncode: int
    stdout: str
    stderr: str


class CliBackendError(RuntimeError):
    """A CLI call failed. ``retryable`` drives the retry ladder;
    ``status_code`` mirrors the HTTP semantics utils._is_unrecoverable
    understands (401 = not logged in)."""

    def __init__(self, message, status_code=None, retryable=True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def strip_fences(text: str) -> str:
    text = text or ""
    match = _FENCE.match(text)
    return (match.group(1) if match else text).strip()


def looks_transient(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in _TRANSIENT_MARKERS)


async def run_subprocess(cmd: list[str], stdin_text: str, timeout: float) -> CliResult:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8")), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise CliBackendError(f"{cmd[0]} timed out after {timeout}s")
    return CliResult(proc.returncode, out.decode("utf-8", "replace"),
                     err.decode("utf-8", "replace"))


class CliBackend:
    name = "cli"

    def __init__(self, model: str, concurrency: int = 3, run=None,
                 timeout: float = 600, max_retries: int = 5,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.model = model
        self.concurrency = concurrency
        self._run = run or run_subprocess
        self.timeout = timeout
        self.max_retries = max_retries
        self.system_prompt = system_prompt
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    # ── subclass hooks ──
    def build_command(self, workdir: str) -> list[str]:
        raise NotImplementedError

    def stdin_text(self, prompt: str) -> str:
        return prompt

    def parse_output(self, result: CliResult, workdir: str) -> str:
        raise NotImplementedError

    # ── machinery ──
    def _semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        sem = self._semaphores.get(loop_id)
        if sem is None:
            sem = self._semaphores[loop_id] = asyncio.Semaphore(self.concurrency)
        return sem

    async def _once(self, prompt: str) -> str:
        async with self._semaphore():
            with tempfile.TemporaryDirectory(prefix="pageindex-cli-") as workdir:
                result = await self._run(self.build_command(workdir),
                                         self.stdin_text(prompt), self.timeout)
                return strip_fences(self.parse_output(result, workdir))

    async def acomplete(self, prompt: str) -> str:
        from ..utils import LLMRetriesExhausted
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await self._once(prompt)
            except CliBackendError as exc:
                last = exc
                if not exc.retryable:
                    raise
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(min(60, 2 ** (attempt + 1)))
        # A status-code-less exhausted ladder (the common shape of a sustained
        # subscription rate/usage-limit failure) must abort the run rather than
        # be treated as an ordinary per-node failure: utils._is_unrecoverable
        # already treats LLMRetriesExhausted as fatal for anything but a 400,
        # unlike CliBackendError which only status codes 401/403/404 make fatal.
        raise LLMRetriesExhausted(
            f"{self.name} ({self.model}) failed after {self.max_retries} attempts: {last}",
            status_code=getattr(last, "status_code", None))

    def complete(self, prompt: str) -> str:
        from ..utils import run_off_loop
        return run_off_loop(asyncio.run, self.acomplete(prompt))
