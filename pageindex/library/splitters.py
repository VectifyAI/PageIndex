"""Profile-specific structure builders that need no LLM.

diary: books whose body is a sequence of dated entries
("12 March 1981 – Thursday" on its own line). Flash sees only a few of these
as headings and leaves a hundreds-of-pages leaf. We rebuild that span as
Year › Month nodes; the entry dates become the month's key_items."""
from __future__ import annotations

import re

MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")
DATE_HEADING = re.compile(
    r"^[ \t]*(?P<day>\d{1,2})[ \t]+(?P<month>" + "|".join(MONTHS) + r")[ \t]+(?P<year>\d{4})"
    r"[ \t]*[–—-][ \t]*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)[ \t]*$",
    re.M)
MIN_ENTRIES = 3


def find_dated_entries(page_texts: list[str], first_page: int = 1) -> list[dict]:
    entries = []
    for offset, text in enumerate(page_texts):
        for match in DATE_HEADING.finditer(text or ""):
            entries.append({
                "title": f"{int(match['day'])} {match['month']} {match['year']}",
                "page": first_page + offset,
                "year": int(match["year"]),
                "month": MONTHS.index(match["month"]) + 1,
            })
    return entries


def build_diary_structure(entries: list[dict], end_page: int) -> list[dict]:
    """Year nodes containing Month leaves. A month runs from its first entry's
    page to the following month's start page — consecutive months deliberately
    share their boundary page, so a diary entry that spills onto the page
    where the next month's first entry also begins is not lost; the last
    month runs to end_page."""
    months: list[dict] = []
    for entry in entries:
        key = (entry["year"], entry["month"])
        if not months or months[-1]["_key"] != key:
            months.append({"_key": key, "title": f"{MONTHS[entry['month'] - 1]} {entry['year']}",
                           "start_index": entry["page"], "end_index": end_page,
                           "key_items": [], "nodes": []})
        months[-1]["key_items"].append(entry["title"])
    for current, following in zip(months, months[1:]):
        current["end_index"] = following["start_index"]
    years: list[dict] = []
    for month in months:
        year = month["_key"][0]
        if not years or years[-1]["title"] != str(year):
            years.append({"title": str(year), "start_index": month["start_index"],
                          "end_index": month["end_index"], "nodes": []})
        years[-1]["nodes"].append(month)
        years[-1]["end_index"] = month["end_index"]
    for month in months:
        month.pop("_key")
    return years


def apply_diary_profile(structure: list[dict], page_texts: list[str]) -> list[dict]:
    entries = find_dated_entries(page_texts)
    if len(entries) < MIN_ENTRIES:
        return structure
    first, last = entries[0]["page"], entries[-1]["page"]
    # back matter: top-level nodes starting after the last entry page
    back = [n for n in structure if n["start_index"] > last]
    diary_end = min(n["start_index"] for n in back) - 1 if back else len(page_texts)
    front = []
    for node in structure:
        if node["start_index"] >= first:
            continue
        clipped = {**node, "end_index": min(node["end_index"], max(node["start_index"], first - 1)),
                   "nodes": []}
        front.append(clipped)
    return front + build_diary_structure(entries, diary_end) + back
