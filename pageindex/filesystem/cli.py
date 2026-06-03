from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Iterator, TextIO

from .agent import (
    PIFSAgentSession,
    REASONING_EFFORT_CHOICES,
    REASONING_SUMMARY_CHOICES,
    run_pifs_agent,
)
from .commands import PIFSCommandError, PIFSCommandExecutor
from .core import PageIndexFileSystem


AGENT_STREAM_MODE_CHOICES = ("off", "tools", "model", "all")
DEFAULT_AGENT_MODEL = "gpt-5.4"
EXIT_COMMANDS = {"exit", "quit", ":q"}
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|.)")
PIFS_CONFIG_FILE_ENV = "PIFS_CONFIG_FILE"
PIFS_WORKSPACE_ENV = "PIFS_WORKSPACE"


def _config_path() -> Path:
    override = os.environ.get(PIFS_CONFIG_FILE_ENV)
    if override:
        return Path(override).expanduser()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "pageindex" / "pifs.json"


def _read_config() -> dict[str, str]:
    path = _config_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid PIFS config file: {path}")
    return {str(key): str(value) for key, value in payload.items() if value is not None}


def _write_config(config: dict[str, str]) -> Path:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _configured_workspace() -> str | None:
    return _read_config().get("workspace")


def _resolve_workspace(value: str | None) -> str | None:
    return value or os.environ.get(PIFS_WORKSPACE_ENV) or _configured_workspace()


def _load_env_file(path: str | None = None, *, workspace: str | None = None) -> Path | None:
    from dotenv import load_dotenv

    if path:
        env_path = Path(path).expanduser()
        if not env_path.exists():
            raise FileNotFoundError(f"env file not found: {env_path}")
        load_dotenv(env_path, override=True)
        return env_path

    env_override = os.environ.get("PIFS_ENV_FILE")
    if env_override:
        return _load_env_file(env_override)

    starts = [Path.cwd()]
    if workspace:
        starts.append(Path(workspace).expanduser())
    seen: set[Path] = set()
    for start in starts:
        current = start.resolve() if start.exists() else start.resolve(strict=False)
        if current.is_file():
            current = current.parent
        for parent in (current, *current.parents):
            candidate = parent / ".env"
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                load_dotenv(candidate, override=False)
                return candidate
    return None


def _agent_model_default() -> str:
    return (
        os.environ.get("PIFS_AGENT_MODEL")
        or os.environ.get("PIFS_MODEL")
        or DEFAULT_AGENT_MODEL
    )


def _add_agent_arguments(
    parser: argparse.ArgumentParser,
    *,
    workspace_default: str | None,
    default_stream_mode: str,
) -> None:
    parser.add_argument("--workspace", default=workspace_default)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--model", default=_agent_model_default())
    parser.add_argument(
        "--stream-mode",
        default=default_stream_mode,
        choices=AGENT_STREAM_MODE_CHOICES,
    )
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=None,
    )
    parser.add_argument(
        "--reasoning-summary",
        choices=REASONING_SUMMARY_CHOICES,
        default=None,
    )


def _parse_agent_command(
    command_name: str,
    argv: list[str],
    *,
    workspace_default: str | None,
    default_stream_mode: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"pifs {command_name}",
        description=f"PageIndex FileSystem {command_name}",
    )
    _add_agent_arguments(
        parser,
        workspace_default=workspace_default,
        default_stream_mode=default_stream_mode,
    )
    if command_name == "ask":
        parser.add_argument("question", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    _load_env_file(args.env_file, workspace=args.workspace)
    args.workspace = _resolve_workspace(args.workspace)
    if not args.workspace:
        parser.error("--workspace is required unless PIFS_WORKSPACE is set or `pifs set workspace <path>` has been run")
    return args


def _filesystem_from_workspace(workspace: str) -> PageIndexFileSystem:
    return PageIndexFileSystem(Path(workspace).expanduser())


def _agent_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.model,
        "stream_mode": args.stream_mode,
        "max_turns": args.max_turns,
        "max_seconds": args.max_seconds,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_summary": args.reasoning_summary,
    }


def _sanitize_chat_question(raw: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", raw)
    chars: list[str] = []
    for char in text:
        if char in {"\b", "\x7f"}:
            if chars:
                chars.pop()
            continue
        if char in {"\r", "\n"}:
            continue
        if ord(char) < 32 or ord(char) == 127:
            continue
        chars.append(char)
    return "".join(chars).strip()


@contextlib.contextmanager
def _suppress_tty_input_echo(stdin: TextIO | None = None) -> Iterator[None]:
    stream = sys.stdin if stdin is None else stdin
    if not hasattr(stream, "isatty") or not stream.isatty():
        yield
        return
    try:
        import termios

        fd = stream.fileno()
        original = termios.tcgetattr(fd)
        muted = original[:]
        muted[3] = muted[3] & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, muted)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            termios.tcflush(fd, termios.TCIFLUSH)
        with contextlib.suppress(Exception):
            termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _run_ask(argv: list[str], *, workspace_default: str | None) -> int:
    args = _parse_agent_command(
        "ask",
        argv,
        workspace_default=workspace_default,
        default_stream_mode="off",
    )
    question_tokens = [token for token in args.question if token != "--"]
    question = " ".join(question_tokens).strip()
    if not question:
        raise ValueError("ask requires a question")

    filesystem = _filesystem_from_workspace(args.workspace)
    answer = run_pifs_agent(filesystem, question, **_agent_kwargs(args))
    if args.stream_mode == "off":
        print(answer)
    return 0


def _run_chat(argv: list[str], *, workspace_default: str | None) -> int:
    args = _parse_agent_command(
        "chat",
        argv,
        workspace_default=workspace_default,
        default_stream_mode="all",
    )
    filesystem = _filesystem_from_workspace(args.workspace)
    session = PIFSAgentSession(filesystem, **_agent_kwargs(args))
    while True:
        try:
            question = _sanitize_chat_question(input("pifs> "))
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break
        with _suppress_tty_input_echo():
            answer = session.run(question)
        if args.stream_mode == "off":
            print(answer)
    return 0


def _run_passthrough(
    command_tokens: list[str],
    *,
    workspace: str,
    json_output: bool,
) -> int:
    filesystem = _filesystem_from_workspace(workspace)
    executor = PIFSCommandExecutor(filesystem, json_output=json_output)
    command = " ".join(shlex.quote(token) for token in command_tokens)
    print(executor.execute(command))
    return 0


def _run_add(argv: list[str], *, workspace: str) -> int:
    parser = argparse.ArgumentParser(
        prog="pifs add",
        description="Add a local file to a PageIndex FileSystem workspace",
    )
    parser.add_argument("physical_path")
    parser.add_argument("virtual_target")
    args = parser.parse_args(argv)

    filesystem = _filesystem_from_workspace(workspace)
    info = filesystem.add_file(args.physical_path, args.virtual_target)
    print(f"added: {info.get('path')}")
    print(f"file_ref: {info['file_ref']}")
    return 0


def _run_set(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="pifs set",
        description="Set PageIndex FileSystem CLI defaults",
    )
    parser.add_argument("name", choices=["workspace"])
    parser.add_argument("value")
    args = parser.parse_args(argv)

    config = _read_config()
    if args.name == "workspace":
        workspace = Path(args.value).expanduser().resolve(strict=False)
        config["workspace"] = str(workspace)
        path = _write_config(config)
        print(f"workspace: {workspace}")
        print(f"config: {path}")
        return 0
    raise ValueError(f"unknown config key: {args.name}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _load_env_file()
    parser = argparse.ArgumentParser(description="PageIndex FileSystem CLI")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    _load_env_file(args.env_file, workspace=args.workspace)
    args.workspace = _resolve_workspace(args.workspace)

    command_tokens = [token for token in args.command if token != "--"]
    json_output = args.json_output

    if not command_tokens:
        parser.error("a filesystem command is required")

    try:
        command_name = command_tokens[0]
        command_args = command_tokens[1:]
        if command_name == "set":
            return _run_set(command_args)
        if command_name == "ask":
            return _run_ask(command_args, workspace_default=args.workspace)
        if command_name == "chat":
            return _run_chat(command_args, workspace_default=args.workspace)
        if command_name == "add":
            if not args.workspace:
                parser.error("--workspace is required unless PIFS_WORKSPACE is set or `pifs set workspace <path>` has been run")
            return _run_add(command_args, workspace=args.workspace)

        if "--json" in command_tokens:
            command_tokens = [token for token in command_tokens if token != "--json"]
            json_output = True
        if not args.workspace:
            parser.error("--workspace is required unless PIFS_WORKSPACE is set or `pifs set workspace <path>` has been run")
        return _run_passthrough(
            command_tokens,
            workspace=args.workspace,
            json_output=json_output,
        )
    except PIFSCommandError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
