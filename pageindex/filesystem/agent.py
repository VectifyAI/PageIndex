from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import re
import sys
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, TextIO

from .commands import PIFSCommandError, PIFSCommandExecutor
from .core import PageIndexFileSystem


TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
PIFS_AGENT_TRACING_ENV = "PAGEINDEX_PIFS_AGENT_TRACING"
PIFS_AGENT_RAW_REASONING_ENV = "PAGEINDEX_PIFS_AGENT_RAW_REASONING"

AGENT_SYSTEM_PROMPT = """
You are the PageIndex FileSystem Demo Agent, developed by the VectifyAI Team.
Your job is to answer questions about the caller's current PageIndex FileSystem
workspace.

You can inspect the corpus only by calling the bash tool. The bash tool is a
read-only PageIndex virtual shell, not a real operating-system shell.

If the user asks who you are, answer with this identity and mention that you can
help inspect and answer questions about the current PIFS workspace. If the user
asks a general question unrelated to the current workspace, do not answer it as
a general-purpose assistant; briefly say that you can only answer workspace-
related questions and invite them to ask about files, folders, metadata, or
document contents in the workspace.

If the user asks what tools or capabilities you have, describe only the PIFS
virtual shell capabilities available inside this workspace: ls, tree, find,
stat, grep, cat, and semantic search commands such as search-summary when they
are available. Do not mention host runtime tools, SDK internals, or orchestration
helpers that are not part of the PIFS shell.

If the user asks a workspace-related topic question without naming a specific
file, treat it as a retrieval task. Use available PIFS discovery commands to
look for relevant files and inspect evidence before answering. Ask the user to
clarify only after a reasonable search cannot identify relevant evidence.
Do not conclude that no relevant document exists from one failed grep. If grep
returns no matches for a workspace topic, verify with available semantic
candidate discovery such as search-summary, or inspect likely document
structure, before saying that the workspace lacks evidence.

Follow the task prompt for command policy, retrieval strategy, and answer
format. If the caller needs stricter behavior, pass an explicit system_prompt.
"""

BASH_TOOL_DESCRIPTION = """
Run a command in the PageIndex FileSystem virtual shell. This is not a real
operating-system shell. By default the tool is read-only: use ls, tree, find,
grep, cat, stat, head, tail, sed, and any dynamically available semantic search
commands described in the workspace context. grep -R is lexical evidence search;
grep does not support regex alternation such as "a|b"; run multiple grep
commands or use search-summary for semantic candidate discovery instead.
semantic search commands such as search-summary return candidate documents and
do not guarantee literal text matches. Use search-summary when the user asks for
summary search, semantic search, or vector search and the command is listed as
available. Quote multi-word semantic queries, for example:
search-summary "Federal Reserve" /documents. Do not write
search-summary Federal Reserve /documents. Errors are returned as text prefixed
with ERROR. Do not call
commands that are not listed as available. When evidence is required, inspect it
with cat or grep before answering. Prefer shell-like target-first cat syntax
with stable targets: cat <path> --structure, cat <path> --page 31-59, and
cat <path> --node 0009. You may also use file_ref or document_id when a path is
ambiguous. After structure identifies a relevant section node, prefer
cat <path> --node <node_id>; use cat <path> --page <range> when the user asks
for page-level evidence, no suitable node exists, or exact page text is needed.
cat <path> --structure is paginated; request more with --offset if needed. Page
reads are limited to three pages at once, node reads to at most five node ids,
and text cat --all returns only the first page of text lines.
For questions about metadata fields, available summaries, or whether metadata
was provided, inspect stat --schema and stat <target> before making claims.
Do not use stat as a general content/topic discovery step. For document Q&A,
prefer search-summary/find/grep for candidates, then cat --structure and
cat --node or cat --page for evidence.
"""

AGENT_TOOL_POLICY = """
Tool policy:
- The bash tool is a PageIndex virtual shell, not an operating-system shell.
- The default agent tool surface is read-only.
- Use only commands listed in the workspace capabilities.
- Folder paths such as /documents are positional command targets; never put folder paths in --where.
- Use --where only with metadata fields shown by stat --schema.
- grep -R performs lexical evidence search.
- grep does not support regex alternation such as "a|b"; run separate grep commands or use search-summary for semantic candidate discovery.
- Semantic search commands are candidate-discovery tools and do not guarantee literal text matches.
- A single failed grep is not enough evidence to say there is no relevant document. If grep returns no matches for a workspace-topic question, verify with search-summary or another available semantic/vector candidate command, or inspect likely document structure, before answering no-evidence.
- If search-summary is available and the user asks for summary search, semantic search, vector search, or "用 summary 搜", use search-summary "<query>" <folder>; quote multi-word queries, for example search-summary "Federal Reserve" /documents; do not translate that request into find --where.
- Tool errors are returned as ERROR text; recover by trying an available command.
- Use cat or grep to gather evidence before making source-backed claims.
- For broad topic, method, or "what solution" questions that are likely about the workspace, search for candidate documents before asking the user to choose a document.
- Use stat only for metadata/schema/status questions or to resolve ambiguous target identity. Do not run stat merely to understand what a document says.
- Prefer target-first cat syntax with stable targets: cat <path> --structure, cat <path> --page 31-59, cat <path> --node <node_id>.
- cat <target> --structure returns at most 25 nodes; use --offset and --limit for more structure pages.
- cat <target> --page accepts at most 3 pages at once. If a larger range is needed, first inspect cat <target> --structure and then read a smaller page range or node.
- cat <target> --node accepts at most 5 node ids at once. Prefer one relevant node when possible.
- cat <target> --all returns at most 100 text lines; use cat <target> --range <start>-<end> for the next page.
- After cat <target> --structure finds a relevant section/subsection with a node_id, prefer cat <target> --node <node_id> for content from that semantic unit.
- Use cat <target> --page <start>-<end> when the user explicitly asks for pages/page ranges, when no suitable node_id exists, or when you need exact page text to verify page-level evidence.
- Avoid fetching a broad page span after a matching node is available unless page-level citation or verification is required.
- Do not call cat --page <target> <start> <end>; if you need a page span, use cat <target> --page <start>-<end>.
- For metadata or summary-field questions, run stat --schema and stat <target> for relevant files before answering; do not infer metadata presence or absence from ls/find output alone.
- Distinguish default/register metadata from caller-provided custom metadata when the evidence supports it.
"""

STREAM_MODE_ALIASES = {
    "": "off",
    "none": "off",
    "false": "off",
    "0": "off",
    "off": "off",
    "tool": "tools",
    "tools": "tools",
    "model": "model",
    "output": "model",
    "outputs": "model",
    "think": "model",
    "all": "all",
    "debug": "all",
}
AGENT_STREAM_MODE_CHOICES = sorted(item for item in STREAM_MODE_ALIASES if item)
REASONING_EFFORT_CHOICES = ["none", "minimal", "low", "medium", "high", "xhigh"]
REASONING_SUMMARY_CHOICES = ["none", "auto", "concise", "detailed"]


def should_use_openai_compatible_chat_model(base_url: str | None) -> bool:
    if not base_url:
        return False
    normalized = base_url.strip().rstrip("/")
    return normalized not in {"https://api.openai.com", "https://api.openai.com/v1"}


def env_flag_enabled(name: str, environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    value = source.get(name, "")
    return value.strip().lower() in TRUTHY_ENV_VALUES


def pifs_agent_tracing_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return env_flag_enabled(PIFS_AGENT_TRACING_ENV, environ)


def should_disable_pifs_agent_tracing(environ: Mapping[str, str] | None = None) -> bool:
    return not pifs_agent_tracing_enabled(environ)


def pifs_agent_raw_reasoning_enabled(environ: Mapping[str, str] | None = None) -> bool:
    return env_flag_enabled(PIFS_AGENT_RAW_REASONING_ENV, environ)


def normalize_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None or not reasoning_effort.strip():
        return None
    effort = reasoning_effort.strip().lower()
    if effort not in REASONING_EFFORT_CHOICES:
        allowed = ", ".join(REASONING_EFFORT_CHOICES)
        raise ValueError(f"Unknown reasoning effort: {reasoning_effort!r}. Allowed: {allowed}")
    return effort


def normalize_reasoning_summary(reasoning_summary: str | None) -> str | None:
    if reasoning_summary is None or not reasoning_summary.strip():
        return None
    summary = reasoning_summary.strip().lower()
    if summary not in REASONING_SUMMARY_CHOICES:
        allowed = ", ".join(REASONING_SUMMARY_CHOICES)
        raise ValueError(f"Unknown reasoning summary: {reasoning_summary!r}. Allowed: {allowed}")
    return None if summary == "none" else summary


def build_agent_model_settings(
    *,
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
) -> Any | None:
    effort = normalize_reasoning_effort(reasoning_effort)
    summary = normalize_reasoning_summary(reasoning_summary)
    if effort is None and summary is None:
        return None
    if effort not in {None, "none"} and summary is None:
        summary = "auto"

    from agents import ModelSettings
    from openai.types.shared import Reasoning

    reasoning_kwargs = {}
    if effort is not None:
        reasoning_kwargs["effort"] = effort
    if summary is not None:
        reasoning_kwargs["summary"] = summary
    return ModelSettings(reasoning=Reasoning(**reasoning_kwargs), verbosity="low")


def normalize_agent_stream_mode(stream_mode: str | None) -> str:
    mode = STREAM_MODE_ALIASES.get((stream_mode or "off").strip().lower())
    if mode is None:
        allowed = ", ".join(sorted({"off", "tools", "model", "all"}))
        raise ValueError(f"Unknown PIFS agent stream mode: {stream_mode!r}. Allowed: {allowed}")
    return mode


def serialize_agent_final_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    if is_dataclass(value):
        return json.dumps(asdict(value), ensure_ascii=False)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def compact_tool_output_preview(
    output: str,
    *,
    preview_chars: int = 700,
    max_lines: int = 8,
) -> str:
    cleaned = str(output).replace("\r", "\n").replace("\f", "\n")
    cleaned = "".join(
        ch if ch == "\n" or ch == "\t" or ord(ch) >= 32 else " "
        for ch in cleaned
    )
    lines = [
        re.sub(r"[ \t]{2,}", " ", line).strip()
        for line in cleaned.splitlines()
        if line.strip()
    ]
    is_large_result = len(cleaned) > preview_chars or len(lines) > max_lines
    preview = "\n".join(lines[:max_lines])
    if len(preview) > preview_chars:
        preview = preview[:preview_chars].rstrip() + "..."
    omitted = len(lines) - min(len(lines), max_lines)
    if is_large_result:
        preview = f"[large PIFS result: {len(cleaned)} chars; showing compact preview]\n" + preview
    if omitted > 0:
        preview += f"\n... [{omitted} more lines omitted from preview]"
    if len(cleaned) > preview_chars:
        preview += "\n... [full result returned to agent; terminal preview shortened]"
    return preview


def build_agent_initial_context(
    filesystem: PageIndexFileSystem,
    *,
    root: str = "/",
    executor: PIFSCommandExecutor | None = None,
    query_context: str | None = None,
) -> str:
    executor = executor or PIFSCommandExecutor(
        filesystem,
        json_output=False,
        query_context=query_context,
    )
    schema = filesystem._metadata_schema()
    schema_fields = schema.get("fields", {})
    schema_sample = dict(list(schema_fields.items())[:50])
    return "\n".join(
        [
            f"Root path: {root}",
            "Top-level listing:",
            executor.execute(f"ls {root}"),
            "Metadata schema summary:",
            json.dumps(
                {
                    "field_count": len(schema_fields),
                    "sample_fields": schema_sample,
                },
                ensure_ascii=False,
            ),
            "Workspace retrieval capabilities:",
            executor.describe_available_command_surfaces(),
        ]
    )


def build_pifs_agent_instructions(
    filesystem: PageIndexFileSystem,
    *,
    root: str = "/",
    system_prompt: str | None = None,
    executor: PIFSCommandExecutor | None = None,
    query_context: str | None = None,
) -> str:
    initial_context = build_agent_initial_context(
        filesystem,
        root=root,
        executor=executor,
        query_context=query_context,
    )
    return "\n\n".join(
        [
            (system_prompt or AGENT_SYSTEM_PROMPT).strip(),
            AGENT_TOOL_POLICY.strip(),
            "Workspace context:\n" + initial_context,
        ]
    )


class PIFSAgentStreamObserver:
    def __init__(
        self,
        stream_mode: str,
        *,
        stream_log: list[dict[str, Any]] | None = None,
        output: TextIO | None = None,
        include_raw_reasoning: bool | None = None,
    ) -> None:
        self.stream_mode = normalize_agent_stream_mode(stream_mode)
        self.stream_log = stream_log
        self.output = output or sys.stdout
        self.include_raw_reasoning = (
            pifs_agent_raw_reasoning_enabled()
            if include_raw_reasoning is None
            else include_raw_reasoning
        )
        self._printed_section: str | None = None
        self._buffers: dict[str, list[str]] = {
            "output": [],
            "think": [],
            "think_summary": [],
            "tool_args": [],
        }

    @property
    def wants_model_stream(self) -> bool:
        return self.stream_mode in {"model", "all"}

    @property
    def wants_tool_stream(self) -> bool:
        return self.stream_mode in {"tools", "all"}

    @property
    def has_output_text(self) -> bool:
        return bool(self._buffers["output"])

    def handle_event(self, event: Any) -> None:
        if getattr(event, "type", None) == "raw_response_event":
            self._handle_raw_response_event(getattr(event, "data", None))
        elif getattr(event, "type", None) == "run_item_stream_event":
            self._handle_run_item_event(event)

    def finish(self, final_output: Any = None) -> None:
        if self.wants_model_stream and not self.has_output_text and final_output:
            self._emit("output", str(final_output), "[llm final output stream]")
        if self._printed_section is not None:
            print(file=self.output, flush=True)
            self._printed_section = None
        if self.stream_log is not None:
            for kind, parts in self._buffers.items():
                text = "".join(parts)
                if text:
                    self.stream_log.append({"kind": kind, "text": text})

    def _handle_raw_response_event(self, data: Any) -> None:
        event_type = getattr(data, "type", "")
        delta = getattr(data, "delta", None)
        if not isinstance(delta, str) or not delta:
            return
        if event_type == "response.output_text.delta":
            self._emit("output", delta, "[llm final output stream]")
        elif event_type == "response.reasoning_text.delta":
            if self.include_raw_reasoning:
                self._emit("think", delta, "[llm reasoning text stream]")
        elif event_type == "response.reasoning_summary_text.delta":
            self._emit("think_summary", delta, "[llm reasoning summary stream]")
        elif event_type == "response.function_call_arguments.delta":
            self._buffers["tool_args"].append(delta)

    def _handle_run_item_event(self, event: Any) -> None:
        name = getattr(event, "name", "")
        item = getattr(event, "item", None)
        item_type = getattr(item, "type", "")
        if self.stream_log is not None and name in {"message_output_created", "reasoning_item_created"}:
            self.stream_log.append({"kind": "run_item", "name": name, "item_type": item_type})

    def _emit(self, kind: str, text: str, label: str) -> None:
        if kind == "tool_args":
            should_print = self.wants_tool_stream
        else:
            should_print = self.wants_model_stream
        if not should_print:
            return
        self._buffers[kind].append(text)
        if self._printed_section != kind:
            if self._printed_section is not None:
                print(file=self.output, flush=True)
            print(f"\n{label}", file=self.output, flush=True)
            self._printed_section = kind
        print(text, end="", file=self.output, flush=True)

    def emit_tool_call(self, command: str, *, force: bool = False) -> None:
        if not command.strip():
            return
        if self.stream_log is not None:
            self.stream_log.append({"kind": "tool_call", "command": command})
        if not (force or self.wants_tool_stream):
            return
        self._start_section("tool_call", "[llm -> pifs command]")
        print(command, file=self.output, flush=True)

    def emit_tool_result(
        self,
        *,
        ok: bool,
        output: str,
        seconds: float,
        force: bool = False,
        preview_chars: int = 1000,
    ) -> None:
        if self.stream_log is not None:
            self.stream_log.append(
                {
                    "kind": "tool_result",
                    "ok": ok,
                    "seconds": round(seconds, 4),
                    "output_chars": len(output),
                    "preview": compact_tool_output_preview(output, preview_chars=preview_chars),
                }
            )
        if not (force or self.wants_tool_stream):
            return
        preview = compact_tool_output_preview(output, preview_chars=preview_chars)
        self._start_section("tool_result", "[pifs -> llm result preview]")
        print(
            f"ok={str(ok).lower()} seconds={seconds:.4f} output_chars={len(output)}",
            file=self.output,
            flush=True,
        )
        print(preview, file=self.output, flush=True)

    def _start_section(self, kind: str, label: str) -> None:
        if self._printed_section is not None:
            print(file=self.output, flush=True)
        print(f"\n{label}", file=self.output, flush=True)
        self._printed_section = kind


def run_pifs_agent(
    filesystem: PageIndexFileSystem,
    question: str,
    *,
    model: str,
    root: str = "/",
    system_prompt: str | None = None,
    max_turns: int = 20,
    max_seconds: float | None = 60,
    verbose: bool = False,
    stream_mode: str = "off",
    reasoning_effort: str | None = None,
    reasoning_summary: str | None = None,
    output_type: type[Any] | None = None,
    tool_log: list[dict[str, Any]] | None = None,
    agent_log: list[dict[str, Any]] | None = None,
) -> str:
    session = PIFSAgentSession(
        filesystem,
        model=model,
        root=root,
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_seconds=max_seconds,
        verbose=verbose,
        stream_mode=stream_mode,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        output_type=output_type,
        tool_log=tool_log,
        agent_log=agent_log,
        persist_conversation=False,
    )
    return session.run(question)


class PIFSAgentSession:
    def __init__(
        self,
        filesystem: PageIndexFileSystem,
        *,
        model: str,
        root: str = "/",
        system_prompt: str | None = None,
        max_turns: int = 20,
        max_seconds: float | None = 60,
        verbose: bool = False,
        stream_mode: str = "off",
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        output_type: type[Any] | None = None,
        tool_log: list[dict[str, Any]] | None = None,
        agent_log: list[dict[str, Any]] | None = None,
        persist_conversation: bool = True,
    ) -> None:
        self.filesystem = filesystem
        self.max_turns = max_turns
        self.max_seconds = max_seconds
        self.verbose = verbose
        self.tool_log = tool_log
        self.agent_log = agent_log
        self.normalized_stream_mode = normalize_agent_stream_mode(stream_mode)
        self.observer: PIFSAgentStreamObserver | None = None

        try:
            from agents import (
                Agent,
                OpenAIChatCompletionsModel,
                function_tool,
                set_tracing_disabled,
            )
            from agents.memory import SQLiteSession
            from openai import AsyncOpenAI
        except ModuleNotFoundError as exc:
            if exc.name == "agents":
                raise RuntimeError(
                    "openai-agents is required to run the PageIndex FileSystem agent"
                ) from exc
            raise

        set_tracing_disabled(should_disable_pifs_agent_tracing())
        self.executor = PIFSCommandExecutor(filesystem, json_output=False)
        instructions = build_pifs_agent_instructions(
            filesystem,
            root=root,
            system_prompt=system_prompt,
            executor=self.executor,
        )

        @function_tool(description_override=BASH_TOOL_DESCRIPTION.strip())
        def bash(command: str) -> str:
            """Run an allowed PageIndex FileSystem virtual shell command."""
            return self._run_bash(command)

        model_settings = build_agent_model_settings(
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
        )
        base_url = os.environ.get("OPENAI_BASE_URL")
        model_config = model
        if should_use_openai_compatible_chat_model(base_url):
            model_config = OpenAIChatCompletionsModel(
                model=model,
                openai_client=AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=base_url,
                ),
            )

        agent_kwargs: dict[str, Any] = {
            "name": "PageIndexFileSystem",
            "instructions": instructions,
            "tools": [bash],
            "model": model_config,
        }
        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings
        if output_type is not None:
            agent_kwargs["output_type"] = output_type
        self.agent = Agent(**agent_kwargs)
        self.session = SQLiteSession("pifs-chat") if persist_conversation else None

    def run(self, question: str) -> str:
        self.executor.query_context = extract_agent_question_text(question)
        self.observer = PIFSAgentStreamObserver(
            self.normalized_stream_mode,
            stream_log=self.agent_log,
        )

        async def _run_streamed() -> str:
            from agents import Runner

            streamed_run = Runner.run_streamed(
                self.agent,
                question,
                max_turns=self.max_turns,
                session=self.session,
            )
            final_output = ""
            try:
                async for event in streamed_run.stream_events():
                    self.observer.handle_event(event)
                final_output = serialize_agent_final_output(streamed_run.final_output)
                return final_output
            finally:
                if not final_output and streamed_run.final_output:
                    final_output = serialize_agent_final_output(streamed_run.final_output)
                self.observer.finish(final_output)

        async def _run() -> str:
            if self.max_seconds is None or self.max_seconds <= 0:
                return await _run_streamed()
            try:
                return await asyncio.wait_for(_run_streamed(), timeout=self.max_seconds)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"MaxSecondsExceeded: exceeded {self.max_seconds:g}s") from exc

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()

    def _run_bash(self, command: str) -> str:
        started = time.time()
        ok = True
        assert self.observer is not None
        self.observer.emit_tool_call(command, force=self.verbose)
        try:
            output = self.executor.execute(command)
        except PIFSCommandError as exc:
            ok = False
            output = f"ERROR: {exc}"
        seconds = time.time() - started
        if self.tool_log is not None:
            self.tool_log.append(
                {
                    "command": command,
                    "ok": ok,
                    "seconds": round(seconds, 4),
                    "output_chars": len(output),
                    "preview": output[:500],
                }
            )
        self.observer.emit_tool_result(
            ok=ok,
            output=output,
            seconds=seconds,
            force=self.verbose,
        )
        return output


def extract_agent_question_text(prompt: str) -> str:
    for line in str(prompt or "").splitlines():
        if line.startswith("Question:"):
            value = line.split(":", 1)[1].strip()
            if value:
                return value
    return str(prompt or "").strip()
