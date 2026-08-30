"""`codex exec` (OpenAI Codex CLI headless) as a completion backend.

Runs on the ChatGPT subscription login. Codex has no system-prompt flag, so
the system prompt is prepended to the stdin prompt. The last assistant
message is read from the -o file; stdout carries only progress lines."""
from __future__ import annotations

import os

from .base import (CliBackend, CliBackendError, CliResult, DEFAULT_SYSTEM_PROMPT,
                   looks_transient)


class CodexCliBackend(CliBackend):
    name = "codex-cli"
    executable = "codex"

    def __init__(self, model: str, concurrency: int = 3, run=None,
                 timeout: float = 600, max_retries: int = 5,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 reasoning_effort: str = "low"):
        super().__init__(model, concurrency=concurrency, run=run, timeout=timeout,
                         max_retries=max_retries, system_prompt=system_prompt)
        self.reasoning_effort = reasoning_effort

    def build_command(self, workdir: str) -> list[str]:
        cmd = [self.executable, "exec", "--ephemeral", "--skip-git-repo-check",
               "-s", "read-only", "-C", workdir,
               "-o", os.path.join(workdir, "last.txt")]
        if self.model:
            cmd += ["-m", self.model]
        cmd += ["-c", f"model_reasoning_effort={self.reasoning_effort}", "-"]
        return cmd

    def stdin_text(self, prompt: str) -> str:
        return f"{self.system_prompt}\n\n{prompt}"

    def parse_output(self, result: CliResult, workdir: str) -> str:
        path = os.path.join(workdir, "last.txt")
        text = ""
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        if not text.strip():
            tail = (result.stderr or result.stdout).strip()[-500:]
            raise CliBackendError(f"codex exited {result.returncode}: {tail}",
                                  retryable=looks_transient(tail))
        return text
