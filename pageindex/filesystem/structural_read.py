from __future__ import annotations

from copy import deepcopy
from typing import Any


def strip_pageindex_text_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_pageindex_text_fields(item) for item in value]
    if isinstance(value, dict):
        return {
            key: strip_pageindex_text_fields(item)
            for key, item in value.items()
            if key != "text"
        }
    return value


def find_pageindex_node(structure: Any, node_id: str) -> dict[str, Any] | None:
    if isinstance(structure, dict):
        if str(structure.get("node_id", "")) == str(node_id):
            return deepcopy(structure)
        for child_key in ("nodes", "children"):
            found = find_pageindex_node(structure.get(child_key), node_id)
            if found is not None:
                return found
    if isinstance(structure, list):
        for item in structure:
            found = find_pageindex_node(item, node_id)
            if found is not None:
                return found
    return None


def first_node_location(node: dict[str, Any]) -> str | None:
    for key in ("line_num", "physical_index", "start_index"):
        value = node.get(key)
        if value is not None and value != "":
            return str(value)
    return None
