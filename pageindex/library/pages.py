"""Page-range helpers shared by summaries, digests and the MCP server.
Pages are 1-based and ranges inclusive, like PageIndex trees."""
from __future__ import annotations

import re

_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+))?\s*$")


def page_text(page_texts: list[str], start: int, end: int) -> str:
    return "\n\n".join(page_texts[i - 1] for i in range(start, end + 1))


def parse_page_spec(spec: str, page_count: int) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        match = _RANGE.match(part)
        if not match:
            raise ValueError(f"Bad page spec {part!r}; use '5', '3,7' or '5-10'")
        lo = int(match.group(1))
        hi = int(match.group(2) or lo)
        if lo < 1 or hi > page_count or lo > hi:
            raise ValueError(f"Pages {part.strip()} outside 1-{page_count}")
        pages.extend(range(lo, hi + 1))
    return pages


def leaf_spans(structure: list[dict]) -> list[tuple[str, int, int]]:
    out = []
    for node in structure:
        children = node.get("nodes") or []
        if children:
            out.extend(leaf_spans(children))
        else:
            out.append((node.get("title", ""), node["start_index"], node["end_index"]))
    return out
