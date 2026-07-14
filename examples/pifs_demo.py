"""A minimal PageIndex FileSystem CLI walkthrough.

Configure the Summary Embedding Profile in your PIFS config and provide its API
key through ``PIFS_EMBEDDING_API_KEY`` before running this example.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(workspace: Path, document: Path, query: str) -> int:
    target = f"/documents/{document.name}"
    commands = [
        ["add", str(document), "/documents"],
        ["tree", "/documents", "-L", "1"],
        ["browse", "/documents", query],
        ["stat", target],
        ["cat", target, "--structure"],
        ["cat", target, "--page", "1"],
        ["grep", query, target],
    ]
    for command in commands:
        print(f"\n$ pifs {' '.join(command)}")
        result = subprocess.run(
            ["pifs", "--workspace", str(workspace), *command],
            check=False,
        )
        if result.returncode:
            return result.returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="PDF or Markdown document")
    parser.add_argument("--workspace", type=Path, default=Path("pifs-workspace"))
    parser.add_argument("--query", default="key findings")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args.workspace, args.document, args.query))
