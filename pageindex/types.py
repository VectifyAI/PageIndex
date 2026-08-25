"""Config shapes for the constructor's ``index=`` / ``chat=`` slots.

The slots are the grouped spelling of the flat constructor arguments —
the same arguments with the side prefix factored out of the names
(``index={"model": ...}`` is ``index_model=``), one spelling per side.
A dict declares its
side by its keys (cloud takes a key, local takes models); an optional
``"mode"`` field states the side explicitly and must agree with the
other keys. The ``"pageindex-cloud"`` string is the label spelling for
"this side is managed", used when the API key lives in the environment.
"""
from __future__ import annotations

from typing import Literal, TypedDict, Union

PAGEINDEX_CLOUD = "pageindex-cloud"


class CloudIndexConfig(TypedDict, total=False):
    """Documents hosted on PageIndex cloud. ``api_key`` may be omitted —
    it then comes from the PAGEINDEX_API_KEY environment variable."""

    mode: Literal["cloud"]
    api_key: str


class LocalIndexConfig(TypedDict, total=False):
    """Documents indexed and stored locally."""

    mode: Literal["local"]
    model: str
    summary_model: str
    backend: dict
    storage_path: str


class ChatConfig(TypedDict, total=False):
    """The chat side: ``mode: "local"`` (or any model/backend key) is
    your own model — the agent runs in your process on your keys;
    ``{"mode": "cloud"}`` alone is the managed chat."""

    mode: Literal["cloud", "local"]
    model: str
    backend: dict


IndexConfig = Union[CloudIndexConfig, LocalIndexConfig]
