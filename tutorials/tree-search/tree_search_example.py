"""Tree search demo over a PageIndex structure.

This script shows how to navigate a PageIndex document tree using an LLM to
prioritise nodes. It can also run a lightweight keyword heuristic when an API
key is not available.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import openai

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pageindex.utils import extract_json


@dataclass
class TreeNode:
    """Represents a PageIndex node with child references."""

    node_id: str
    title: str
    summary: str
    start_index: int
    end_index: int
    depth: int
    path_ids: Tuple[str, ...]
    path_titles: Tuple[str, ...]
    children: List["TreeNode"] = field(default_factory=list)

    def short_summary(self, max_chars: int) -> str:
        if len(self.summary) <= max_chars:
            return self.summary
        clipped = self.summary[:max_chars].rsplit(" ", 1)[0]
        return f"{clipped}..."

    def pretty_path(self) -> str:
        return " > ".join(self.path_titles)


def load_structure(path: str) -> Tuple[List[TreeNode], Dict[str, TreeNode]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    structure = payload.get("structure", [])
    roots: List[TreeNode] = []
    lookup: Dict[str, TreeNode] = {}

    def build(
        node: dict,
        parent_ids: Tuple[str, ...],
        parent_titles: Tuple[str, ...],
        depth: int,
    ) -> TreeNode:
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            node_id = "auto_" + "_".join(parent_ids + (node.get("title", "unknown"),))
        title = node.get("title", "")
        summary = node.get("summary", "")
        start_index = int(node.get("start_index", -1))
        end_index = int(node.get("end_index", -1))
        path_ids = parent_ids + (node_id,)
        safe_title = title or node_id
        path_titles = parent_titles + (safe_title,)
        children_payload = node.get("nodes", [])
        tree_node = TreeNode(
            node_id=node_id,
            title=title,
            summary=summary,
            start_index=start_index,
            end_index=end_index,
            depth=depth,
            path_ids=path_ids,
            path_titles=path_titles,
        )
        lookup[node_id] = tree_node
        for child in children_payload:
            child_node = build(child, path_ids, path_titles, depth + 1)
            tree_node.children.append(child_node)
        return tree_node

    for item in structure:
        roots.append(build(item, tuple(), tuple(), 0))

    return roots, lookup


def render_candidates(nodes: Sequence[TreeNode], max_summary_chars: int) -> str:
    lines: List[str] = []
    for node in nodes:
        header = f"[{node.node_id}] depth={node.depth} pages={node.start_index}-{node.end_index} title={node.title}"
        lines.append(textwrap.fill(header, width=110))
        summary = node.short_summary(max_summary_chars)
        lines.append(textwrap.fill(f"summary: {summary}", width=110, subsequent_indent="    "))
        if node.children:
            child_parts = [f"{child.node_id}:{child.title}" for child in node.children[:8]]
            child_line = "children: " + ", ".join(child_parts)
            lines.append(textwrap.fill(child_line, width=110, subsequent_indent="    "))
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(query: str, candidates: Sequence[TreeNode], max_summary_chars: int) -> str:
    candidate_block = render_candidates(candidates, max_summary_chars)
    prompt = f"""
You are assisting with targeted retrieval over a document hierarchy. A query is
provided together with several candidate nodes from the document tree. Decide
which nodes are useful for answering the query and which ones should be
expanded (i.e. their children examined in future steps).

Query: {query}

Candidate nodes:\n{candidate_block}

Return a JSON object with the following keys:
- "thinking": short reasoning string
- "relevant_nodes": list of objects {{"node_id": str, "reason": str}}
- "expand": list of node_id strings to inspect next (subset of the candidates that have children)
Only reference node_ids that appear in the candidate list. Use [] for empty lists.
"""
    return textwrap.dedent(prompt).strip()


def lexical_score(query_terms: Sequence[str], node: TreeNode) -> float:
    haystack = f"{node.title} {node.summary}".lower()
    hits = sum(1 for term in query_terms if term and term in haystack)
    return hits / max(1, len(query_terms))


def offline_tree_search(query: str, roots: Sequence[TreeNode], *, max_depth: int, branch_factor: int) -> List[Tuple[TreeNode, float, str]]:
    terms = re.findall(r"\w+", query.lower())
    queue: Deque[TreeNode] = deque(roots)
    selected: List[Tuple[TreeNode, float, str]] = []

    while queue:
        node = queue.popleft()
        score = lexical_score(terms, node)
        if score > 0:
            selected.append((node, score, "keyword-match"))
        if node.children and node.depth < max_depth:
            ranked_children = sorted(node.children, key=lambda child: lexical_score(terms, child), reverse=True)
            for child in ranked_children[:branch_factor]:
                queue.append(child)

    selected.sort(key=lambda item: item[1], reverse=True)
    return selected


def run_llm_tree_search(
    *,
    query: str,
    roots: Sequence[TreeNode],
    lookup: Dict[str, TreeNode],
    client: openai.OpenAI,
    model: str,
    max_depth: int,
    branch_factor: int,
    prompt_node_limit: int,
    max_summary_chars: int,
    max_turns: int,
    verbose: bool,
) -> List[Tuple[TreeNode, str]]:
    queue: Deque[TreeNode] = deque(roots)
    seen: set[str] = set()
    collected: List[Tuple[TreeNode, str]] = []
    turns = 0
    query_terms = re.findall(r"\w+", query.lower())

    while queue and turns < max_turns:
        batch: List[TreeNode] = []
        while queue and len(batch) < prompt_node_limit:
            node = queue.popleft()
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            batch.append(node)

        if not batch:
            break

        turns += 1
        candidates = list(batch)
        prompt = build_prompt(query, candidates, max_summary_chars)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You help choose relevant sections from a document tree.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        parsed = extract_json(content)
        if not parsed:
            logging.warning("Failed to parse model response: %s", content)
            continue

        thinking = str(parsed.get("thinking", "")).strip()
        if verbose and thinking:
            logging.info("model thinking: %s", thinking)

        relevant = parsed.get("relevant_nodes", []) or []
        expand_list = parsed.get("expand", []) or []

        for entry in relevant:
            node_id = entry.get("node_id")
            reason = entry.get("reason", "")
            if not node_id or node_id not in lookup:
                continue
            node = lookup[node_id]
            collected.append((node, reason))

        expand_targets = {node_id for node_id in expand_list if node_id in lookup}
        for node_id in expand_targets:
            node = lookup[node_id]
            if node.depth >= max_depth:
                continue
            ranked_children = sorted(
                node.children,
                key=lambda child: lexical_score(query_terms, child),
                reverse=True,
            )
            for child in ranked_children[:branch_factor]:
                if child.node_id not in seen:
                    queue.append(child)

    return collected


def configure_logging(verbose: bool) -> None:
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tree search over a PageIndex structure")
    parser.add_argument("query", help="question to search for")
    parser.add_argument(
        "--structure",
        default="results/Attention_is_all_you_need_structure.json",
        help="path to a PageIndex structure JSON",
    )
    parser.add_argument("--model", default="gpt-4.1-nano-2025-04-14", help="OpenAI model name")
    parser.add_argument("--api-key", dest="api_key", help="OpenAI API key (defaults to env)")
    parser.add_argument("--max-depth", type=int, default=3, help="maximum depth to explore")
    parser.add_argument("--branch-factor", type=int, default=3, help="children to expand per node")
    parser.add_argument("--prompt-node-limit", type=int, default=5, help="candidate nodes per LLM turn")
    parser.add_argument("--max-summary-chars", type=int, default=420, help="summary budget per node in prompt")
    parser.add_argument("--max-turns", type=int, default=6, help="maximum LLM turns")
    parser.add_argument("--offline", action="store_true", help="use keyword heuristic instead of LLM")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args()

    configure_logging(args.verbose)

    roots, lookup = load_structure(args.structure)

    if args.offline:
        results = offline_tree_search(
            args.query,
            roots,
            max_depth=args.max_depth,
            branch_factor=args.branch_factor,
        )
        if not results:
            print("No matching nodes found with keyword heuristic.")
            return
        print("Keyword heuristic results:\n")
        for rank, (node, score, reason) in enumerate(results, start=1):
            print(f"{rank}. [{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index})")
            print(f"   path: {node.pretty_path()}")
            print(f"   score: {score:.3f} ({reason})")
            print(f"   summary: {node.short_summary(200)}\n")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY/CHATGPT_API_KEY or pass --api-key. Alternatively use --offline.")

    client = openai.OpenAI(api_key=api_key)

    results = run_llm_tree_search(
        query=args.query,
        roots=roots,
        lookup=lookup,
        client=client,
        model=args.model,
        max_depth=args.max_depth,
        branch_factor=args.branch_factor,
        prompt_node_limit=args.prompt_node_limit,
        max_summary_chars=args.max_summary_chars,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )

    if not results:
        print("LLM search did not return any nodes. Try increasing max-turns or enabling --offline.")
        return

    print("LLM-guided tree search results:\n")
    seen_ids: set[str] = set()
    for rank, (node, reason) in enumerate(results, start=1):
        if node.node_id in seen_ids:
            continue
        seen_ids.add(node.node_id)
        print(f"{rank}. [{node.node_id}] {node.title} (pages {node.start_index}-{node.end_index})")
        print(f"   path: {node.pretty_path()}")
        if reason:
            print(f"   reason: {reason}")
        print(f"   summary: {node.short_summary(200)}\n")


if __name__ == "__main__":
    main()
