from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

from .commands import PIFSCommandError, PIFSCommandExecutor
from .core import PageIndexFileSystem


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="PageIndex FileSystem CLI")
    parser.add_argument("--workspace", default=os.environ.get("PIFS_WORKSPACE"))
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command_tokens = [token for token in args.command if token != "--"]
    json_output = args.json_output
    if "--json" in command_tokens:
        command_tokens = [token for token in command_tokens if token != "--json"]
        json_output = True

    if not args.workspace:
        parser.error("--workspace is required unless PIFS_WORKSPACE is set")
    if not command_tokens:
        parser.error("a filesystem command is required")

    filesystem = PageIndexFileSystem(Path(args.workspace).expanduser())
    executor = PIFSCommandExecutor(filesystem, json_output=json_output)
    try:
        command = " ".join(shlex.quote(token) for token in command_tokens)
        print(executor.execute(command))
    except PIFSCommandError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
