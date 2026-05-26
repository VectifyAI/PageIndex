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


def flatten_pageindex_structure_nodes(structure: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(value: Any, *, depth: int, parent_node_id: str | None) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, depth=depth, parent_node_id=parent_node_id)
            return
        if not isinstance(value, dict):
            return

        node_id = value.get("node_id")
        child_values: list[Any] = []
        for child_key in ("nodes", "children"):
            children = value.get(child_key)
            if isinstance(children, list):
                child_values.extend(children)

        row = {
            key: strip_pageindex_text_fields(item)
            for key, item in value.items()
            if key not in {"text", "nodes", "children"}
        }
        row["depth"] = depth
        row["children_count"] = len(child_values)
        if parent_node_id:
            row["parent_node_id"] = parent_node_id
        rows.append(row)

        next_parent = str(node_id) if node_id is not None else parent_node_id
        for child in child_values:
            visit(child, depth=depth + 1, parent_node_id=next_parent)

    visit(structure, depth=0, parent_node_id=None)
    return rows


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
