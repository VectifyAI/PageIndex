"""Jev-gated hybrid tree navigation for a document's section tree.

PageIndex's agentic retrieval walks a document's section tree with an LLM
at every fan-out — measured at seconds per level. The replay experiment in
``experiments/jev_retrieval`` measured the TypeSafe System One decision API
("Jev") routing the same decision points in ~0.3 s with strong agreement,
so this layer uses Jev for every routing decision and keeps the SDK's own
LLM for the cases Jev is not trusted with:

- Jev top-1 probability >= ACCEPT — descend the single child.
- top-1 in [EXPAND, ACCEPT) — descend the top-2 children.
- top-1 < EXPAND, or the fan-out exceeds MAX_FANOUT — escalate to the LLM.
- the LLM escalation itself fails — expand all children (safe side).

Jev is a hard dependency, never silently bypassed: a missing key or a Jev
call that failed through the transport's retries raises ``JevUnavailable``
(callers render it as a ``PageIndexAPIError`` / error envelope that names
the configuration fix). Only the SDK's own LLM path may fall back.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

from .errors import PageIndexAPIError

#: The System One decision endpoint and model; hardcoded per plan (cost and
#: usage auditing left for when there is tuning evidence to fund it).
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
JEV_API_KEY_ENV = "TYPESAFE_API_KEY"

_QUESTION = "relevant_child"
_NONE = "NONE"
_NONE_DESCRIPTION = "No relevant child; stop expanding this branch."

logger = logging.getLogger(__name__)

# Below the decision layer, so the model never plays retry loop. Same shape
# as the MCP bridge's policy: read=0 — a read timeout is a full wait the
# server may have acted on, never replayed; Retry-After ignored.
_RETRY = Retry(total=3, read=0, backoff_factor=1,
               status_forcelist=(429, *range(500, 600)),
               allowed_methods=None, raise_on_status=False,
               respect_retry_after_header=False)
_TIMEOUT = (10, 120)

# Jev rounds probabilities to two decimals, so a well-formed distribution
# can legitimately sum to 0.99 or 1.01.
_PROBABILITY_TOLERANCE = 0.015

# Gate and budget constants, module-level on purpose: config.yaml is a
# loud-fail validated surface and no tuning evidence backs new knobs yet.
ACCEPT = 0.7   # P(top1) at or above this — descend the single top child
EXPAND = 0.5   # top-1 in [EXPAND, ACCEPT) — descend the top-2 children
MAX_FANOUT = 64      # above this, skip Jev and ask the LLM directly
MAX_JEV_CALLS = 64   # walk budget: Jev + LLM decisions per route() call
MAX_LLM_CALLS = 8    # LLM escalations per route() call ...
MAX_DEPTH = 12       # ... beyond it, remaining escalations expand all

_SUMMARY_CLIP = 200
_LLM_CHOICE_CAP = 3


class JevUnavailable(PageIndexAPIError):
    """Jev could not be reached or answered, and it is the routing layer's
    required decision maker — so callers must fail, not silently walk the
    tree themselves. A PageIndexAPIError subclass: ``search()`` propagates
    it as-is; the find_pages tool renders it as its error envelope."""


def _env_jev_key() -> str:
    # .env support lives in utils' import-time load_dotenv(); imported here,
    # not at module level, to keep `import pageindex` free of the utils
    # stack (pinned by tests/test_package_surface.py).
    from . import utils  # noqa: F401
    key = os.environ.get(JEV_API_KEY_ENV)
    if not key:
        raise JevUnavailable(
            f"Jev-gated navigation reads the TypeSafe API key from the "
            f"{JEV_API_KEY_ENV} environment variable, which is not set — "
            "export it to enable search() and find_pages().")
    return key


def _clip(text: Any, limit: int = _SUMMARY_CLIP) -> str:
    """A one-line summary snippet for a prompt listing."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())[:limit]


class SystemOneClient:
    """Minimal TypeSafe System One client: one endpoint, choice questions.

    Synchronous, requests-only, one pooled session per instance (routing
    a tree comes in bursts), transport retries below the decision layer.
    """

    def __init__(self, api_key: str, timeout: "tuple[int, int]" = _TIMEOUT):
        self._api_key = api_key
        self._timeout = timeout
        self._session = requests.Session()
        for scheme in ("https://", "http://"):
            self._session.mount(scheme, HTTPAdapter(max_retries=_RETRY))

    def _post(self, payload: dict) -> requests.Response:
        try:
            return self._session.post(
                ENDPOINT, json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout)
        except requests.RequestException as exc:
            # The key lives in a header, never in the exception text.
            raise JevUnavailable(f"Could not reach the Jev API: {exc}") from exc

    def ask_choice(self, state_text: str,
                   criteria: "dict[str, str]") -> "dict[str, float]":
        """One calibrated distribution over ``criteria`` (ids -> description).

        Validates the response shape and the distribution itself (covers
        every criterion, sums to 1 within the two-decimal rounding
        tolerance, argmax consistent with the reported choice) — a
        well-formed-looking but unusable answer is as fatal to a routing
        walk as a transport failure. Raises ``JevUnavailable``.
        """
        payload = {
            "model": MODEL,
            "state": {"text": state_text},
            "questions": {_QUESTION: {"type": "choice",
                                      "criteria": dict(criteria)}},
        }
        response = self._post(payload)
        if response.status_code >= 400:
            raise JevUnavailable(
                f"Jev request failed: HTTP {response.status_code} "
                f"({response.text[:200]})")
        try:
            answer = (response.json()["answers"][_QUESTION])
        except (ValueError, KeyError, TypeError) as exc:
            raise JevUnavailable("Jev returned a response without a "
                                 f"{_QUESTION} answer.") from exc
        return _validate_distribution(answer, criteria)

    def close(self) -> None:
        self._session.close()


def _validate_distribution(answer: Any,
                           criteria: "dict[str, str]") -> "dict[str, float]":
    """The probabilities dict for ``criteria``, or ``JevUnavailable``."""
    try:
        probabilities = answer["probabilities"]
        choice = answer["choice"]
    except (KeyError, TypeError) as exc:
        raise JevUnavailable("Jev answer lacks probabilities or choice.") from exc
    if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
        raise JevUnavailable(
            "Jev probabilities do not cover exactly the question's criteria.")
    values: dict[str, float] = {}
    for name, value in probabilities.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevUnavailable("Jev probabilities contain a non-numeric value.")
        if not 0 <= value <= 1:
            raise JevUnavailable("Jev probabilities contain a value outside [0, 1].")
        values[name] = float(value)
    if abs(sum(values.values()) - 1.0) > _PROBABILITY_TOLERANCE:
        raise JevUnavailable(
            "Jev probabilities do not form a distribution (sum != 1).")
    if choice not in values or values[choice] < max(values.values()):
        raise JevUnavailable("Jev's reported choice is not the argmax of its "
                             "own probabilities.")
    return values


def _index_tree(tree: Any) -> "tuple[dict[str, dict], dict[str, list[str]], list[str]]":
    """Pre-order index of a stored tree: (node_by_id, children_by_id,
    root_ids). Iterative — deep chains must not pay recursion. A node
    without node_id gets a unique synthetic id; a malformed duplicate id
    resolves to its first pre-order occurrence (the walk is keyed by id and
    the visited set makes any id reuse terminate)."""
    node_of: dict[str, dict] = {}
    children_of: dict[str, list[str]] = {}
    root_ids: list[str] = []
    auto = 0
    stack = [(node, None) for node in reversed(list(tree or []))]
    while stack:
        node, parent_id = stack.pop()
        raw = node.get("node_id")
        if raw:
            node_id = str(raw)
        else:
            auto += 1
            node_id = f"#auto-{auto}"
        if parent_id is None:
            root_ids.append(node_id)
        else:
            children_of[parent_id].append(node_id)
        if node_id in node_of:
            continue  # duplicate id: the first occurrence owns the subtree
        node_of[node_id] = node
        children_of[node_id] = []
        for child in reversed(node.get("nodes") or []):
            stack.append((child, node_id))
    return node_of, children_of, root_ids


class JevRouter:
    """The routing kernel: walks a stored tree, asking Jev at each fan-out
    and escalating to the SDK's own LLM when Jev is unconfident or the
    fan-out is too wide for one choice question."""

    def __init__(self, jev: Any, llm_model: Optional[str], **overrides: Any):
        self._jev = jev
        self._llm_model = llm_model
        limits: dict[str, Any] = {
            "ACCEPT": ACCEPT, "EXPAND": EXPAND, "MAX_FANOUT": MAX_FANOUT,
            "MAX_JEV_CALLS": MAX_JEV_CALLS, "MAX_LLM_CALLS": MAX_LLM_CALLS,
            "MAX_DEPTH": MAX_DEPTH,
        }
        unknown = set(overrides) - set(limits)
        if unknown:
            raise ValueError(
                f"Unknown JevRouter overrides: {', '.join(sorted(unknown))}.")
        self._limits = {**limits, **overrides}

    # ── the walk ──

    def route(self, query: str, tree: Any, doc_name: Optional[str] = None,
              ) -> "dict[str, Any]":
        """Navigate ``tree`` for ``query``; returns ``{"hits", "stats"}``.

        ``hits`` are the raw tree nodes whose page span answers the query
        (a truncated branch's node itself — its ``start_index`` still
        covers its descendants). ``stats`` reports ``jev_calls``,
        ``llm_calls``, ``pruned`` and ``fallbacks`` for the envelope's
        ``navigated`` block. Iterative throughout: deep chains must not
        recurse, and the frontier left when the budget runs out joins the
        hits — an over-inclusive bias, never an under-inclusive one.
        """
        stats = {"jev_calls": 0, "llm_calls": 0, "pruned": 0, "fallbacks": 0}
        hits: list[dict] = []
        visited: set[str] = set()
        node_of, children_of, root_ids = _index_tree(tree)
        stack = [(node_id, 0) for node_id in reversed(root_ids)]
        budget = self._limits["MAX_JEV_CALLS"]
        while stack and stats["jev_calls"] + stats["llm_calls"] < budget:
            node_id, depth = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = node_of[node_id]
            children_ids = children_of.get(node_id) or []
            if not children_ids or depth >= self._limits["MAX_DEPTH"]:
                hits.append(node)
                continue
            if len(children_ids) > self._limits["MAX_FANOUT"]:
                # One choice question cannot fairly compare 300 options;
                # the LLM sees the listing as text and narrows it instead.
                chosen = self._llm_pick(query, node, node_of, children_ids,
                                        stats)
            else:
                chosen = self._jev_pick(query, node, node_of, children_ids,
                                        stats, doc_name)
            for child_id in chosen:
                # A decision source naming an id outside this fan-out (a
                # duck-typed jev, a malformed tree) is skipped, not fatal.
                if child_id in node_of:
                    stack.append((child_id, depth + 1))
        # Budget exhausted: the undispatched frontier is hits, not losses.
        hits.extend(node_of[node_id] for node_id, _ in stack
                    if node_id not in visited)
        return {"hits": hits, "stats": stats}

    def _jev_pick(self, query: str, node: dict, node_of: dict,
                  children_ids: list, stats: dict,
                  doc_name: Optional[str]) -> list[str]:
        """The gated Jev decision: NONE prunes, high confidence descends
        alone, the middle band takes the top two, low confidence escalates.
        JevUnavailable propagates — the caller fails, it never silently
        expands."""
        state_lines = [f"Node: {node.get('title', '')}"]
        if doc_name:
            state_lines.insert(0, f"Document: {doc_name}")
        state_text = "\n".join([
            *state_lines,
            f"User query: {query}",
            "Candidate children (id: title):",
            *[f"{child_id}: {node_of[child_id].get('title', '')}"
              for child_id in children_ids],
        ])
        criteria = {child_id: node_of[child_id].get("title", "")
                    for child_id in children_ids}
        criteria[_NONE] = _NONE_DESCRIPTION
        stats["jev_calls"] += 1
        probabilities = self._jev.ask_choice(state_text, criteria)
        top = max(probabilities, key=probabilities.get)
        if top == _NONE:
            stats["pruned"] += 1
            return []
        if probabilities[top] >= self._limits["ACCEPT"]:
            return [top]
        if probabilities[top] >= self._limits["EXPAND"]:
            second = max((n for n in probabilities
                          if n not in (top, _NONE)),
                         key=probabilities.get, default=None)
            return [n for n in (top, second) if n is not None]
        return self._llm_pick(query, node, node_of, children_ids, stats)

    def _llm_pick(self, query: str, node: dict, node_of: dict,
                  children_ids: list, stats: dict) -> list[str]:
        """The SDK's own LLM narrows the children. Returns the chosen ids —
        possibly none, an explicit prune decision. Any failure (retries
        exhausted, unusable output, escalation budget gone) falls back to
        expanding all children: the safe side is the SDK's own model, never
        Jev."""
        if stats["llm_calls"] >= self._limits["MAX_LLM_CALLS"]:
            stats["fallbacks"] += 1
            return list(children_ids)
        stats["llm_calls"] += 1
        listing = "\n".join(
            f"- {child_id}: {node_of[child_id].get('title', '')}"
            + (f" — {_clip(node_of[child_id].get('summary'))}"
               if node_of[child_id].get("summary") else "")
            for child_id in children_ids)
        prompt = (
            "You are locating the answer to a query inside a document's "
            "section tree.\n"
            + (f"Parent section: {node.get('title', '')}\n" if node else "")
            + f"User query: {query}\n\n"
            f"Candidate children:\n{listing}\n\n"
            "Return ONLY a JSON array of the ids of the children that may "
            f"contain the answer, at most {_LLM_CHOICE_CAP} of them, e.g. "
            '[\"0001\", \"0002\"]. Return [] if none could contain it.')
        from .utils import extract_json, llm_completion
        try:
            content = llm_completion(self._llm_model, prompt)
            data = extract_json(content)
            ids = data if isinstance(data, list) else (
                data.get("ids") if isinstance(data, dict) else None)
            if not isinstance(ids, list) or not all(
                    isinstance(one_id, str) for one_id in ids):
                raise ValueError(
                    f"LLM escalation returned unusable output: {content!r}")
        except Exception as exc:
            # Only the SDK's own model may fall back — and every way it
            # can fail (retries exhausted, malformed output, transport)
            # lands on the same safe side: expand, don't abort. Expanding
            # is recall-widening, not an error answer, so this stays a
            # logged warning and a fallbacks count, never a raise.
            logger.warning("Jev router: LLM escalation failed (%s); "
                           "expanding all %d children", exc,
                           len(children_ids))
            stats["fallbacks"] += 1
            return list(children_ids)
        children = set(children_ids)
        seen: list[str] = []
        for one_id in ids:
            if one_id in children and one_id not in seen:
                seen.append(one_id)
            if len(seen) == _LLM_CHOICE_CAP:
                break
        if not seen:
            stats["pruned"] += 1
        return seen


# ── hit shaping (shared by client.search and the find_pages tool) ──

def page_numbers(hits: list) -> list[int]:
    """The hit nodes' start pages, sorted and distinct."""
    return sorted({node["start_index"] for node in hits
                   if isinstance(node.get("start_index"), int)
                   and node["start_index"] >= 1})


def format_ranges(pages: list[int]) -> str:
    """Compress [3,4,5,7] into '3-5,7' — the get_page_content pages syntax.
    Mirrors agent_tools._format_page_spec, kept local: agent_tools imports
    this module."""
    if not pages:
        return ""
    ordered = sorted(set(pages))
    ranges: list[str] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def page_spec(hits: list) -> str:
    """The hits as a page specification string, e.g. '3-5,7' ('' for none)."""
    return format_ranges(page_numbers(hits))


def page_ranges(hits: list) -> list[dict]:
    """The hits as {start, end, title, node_id} rows, ordered by page."""
    rows = [{"start": node.get("start_index"),
             "end": node.get("end_index", node.get("start_index")),
             "title": node.get("title", ""),
             "node_id": node.get("node_id")}
            for node in hits
            if isinstance(node.get("start_index"), int)]
    return sorted(rows, key=lambda row: row["start"])


def build_router(llm_model: Optional[str]) -> JevRouter:
    """A router wired to the live Jev endpoint; raises ``JevUnavailable``
    immediately when ``TYPESAFE_API_KEY`` is missing."""
    return JevRouter(SystemOneClient(_env_jev_key()), llm_model)
