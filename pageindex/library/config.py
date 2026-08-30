"""Library-wide settings: where the store lives and which models to use."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DEFAULT_HOME = "~/github/repos/book-library"
CONFIG_FILE = "library.yaml"


@dataclass
class LibraryConfig:
    home: Path
    index_model: str = "claude-cli/sonnet"
    digest_model: str = "claude-cli/sonnet"
    profile: str = "nonfiction"
    max_leaf_pages: int = 40

    @property
    def storage_path(self) -> Path:
        return self.home / ".pageindex"

    @property
    def digests_dir(self) -> Path:
        return self.home / "digests"

    @classmethod
    def load(cls, home: str | os.PathLike | None = None) -> "LibraryConfig":
        root = Path(home or os.environ.get("BOOKS_HOME") or DEFAULT_HOME).expanduser()
        data = {}
        path = root / CONFIG_FILE
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        allowed = {f.name for f in fields(cls)} - {"home"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        return cls(home=root, **data)
