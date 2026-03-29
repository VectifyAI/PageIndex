"""
CodeSearcher — search indexed code repositories with LLM ranking.
Supports relevance, definition, and impact modes.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import ChatGPT_API

RESULTS_DIR = Path(__file__).parent.parent / "results"
# Primary model via PAGEINDEX_MODEL; fallback chain tried in order on failure.
MODEL = os.getenv("PAGEINDEX_MODEL", "deepseek-v3.2")
_FALLBACK_MODELS = ["deepseek-v3.2", "qwen3-coder", "doubao-seed-2.0"]
API_KEY = os.getenv("CHATGPT_API_KEY")


class CodeSearcher:
    def __init__(self, results_dir: str = None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR

    async def search(
        self,
        query: str,
        repo_names: List[str] = None,
        top_k: int = 5,
        mode: str = "relevance",
    ) -> str:
        indices = self._load_indices(repo_names)
        if not indices:
            return "No code indices found. Run index_code first."

        all_symbols = []
        call_graphs = {}
        for idx in indices:
            call_graphs.update(idx.get("call_graph", {}))
            for fi in idx.get("files", []):
                fp = fi.get("file_path", "")
                for sym in fi.get("symbols", []):
                    sym["_file_path"] = fp
                    sym["_repo"] = idx.get("repo_name", "")
                    all_symbols.append(sym)

        if not all_symbols:
            return "No symbols found in indexed repos."

        if mode == "definition":
            return self._search_definition(query, all_symbols)
        elif mode == "impact":
            return self._search_impact(query, all_symbols, call_graphs)
        else:
            return await self._search_relevance(query, all_symbols, call_graphs, top_k)

    def _load_indices(self, repo_names: Optional[List[str]] = None) -> List[dict]:
        indices = []
        for f in self.results_dir.glob("*_code_index.json"):
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                if repo_names and data.get("repo_name") not in repo_names:
                    continue
                indices.append(data)
            except Exception:
                continue
        return indices

    def _search_definition(self, query: str, symbols: List[dict]) -> str:
        query_lower = query.lower()
        matches = []
        for sym in symbols:
            name = sym.get("name", "")
            if query_lower in name.lower() or name.lower() in query_lower:
                matches.append(sym)

        if not matches:
            return f"No symbol matching '{query}' found."

        lines = [f"## Definition search: {query}\n"]
        for sym in matches[:10]:
            lr = sym.get("line_range", [0, 0])
            lines.append(f"### [{sym['_repo']}] {sym['_file_path']}:{sym['name']}")
            lines.append(f"**Type**: {sym.get('type', '?')} | **Lines**: {lr[0]}-{lr[1]}")
            lines.append(f"```\n{sym.get('signature', '')}\n```")
            if sym.get("docstring"):
                lines.append(f"> {sym['docstring'][:200]}")
            lines.append("")
        return "\n".join(lines)

    def _search_impact(self, query: str, symbols: List[dict], call_graph: Dict) -> str:
        # Find the symbol
        query_lower = query.lower()
        target = None
        for sym in symbols:
            if sym.get("name", "").lower() == query_lower or query_lower in sym.get("name", "").lower():
                target = sym
                break

        if not target:
            return f"Symbol '{query}' not found in index."

        target_key = f"{target['_file_path']}:{target['name']}"
        callers = self._get_callers(target_key, call_graph, depth=3)

        lines = [f"## Impact analysis: {target['name']}\n"]
        lines.append(f"**Defined in**: {target['_file_path']} (lines {target.get('line_range', [0,0])[0]}-{target.get('line_range', [0,0])[1]})")
        if callers:
            lines.append(f"\n**Called by** ({len(callers)} callers):")
            for caller, dist in callers:
                hop = "direct" if dist == 1 else f"{dist}-hop"
                lines.append(f"- `{caller}` ({hop})")
        else:
            lines.append("\nNo callers found in the index.")
        return "\n".join(lines)

    def _get_callers(self, target_key: str, call_graph: Dict, depth: int = 3) -> List[Tuple[str, int]]:
        """BFS to find all callers up to `depth` hops."""
        callers = []
        visited = {target_key}
        frontier = {target_key}

        for d in range(1, depth + 1):
            next_frontier = set()
            for caller_key, callees in call_graph.items():
                if caller_key in visited:
                    continue
                if any(c in frontier for c in callees):
                    callers.append((caller_key, d))
                    next_frontier.add(caller_key)
                    visited.add(caller_key)
            frontier = next_frontier
            if not frontier:
                break

        return callers

    async def _search_relevance(self, query: str, symbols: List[dict], call_graph: Dict, top_k: int) -> str:
        # Stage 1: fast pre-filter by keyword overlap
        candidates = self._prefilter(query, symbols, max_candidates=30)

        if not candidates:
            return f"No relevant symbols found for: {query}"

        # Stage 2: LLM ranking
        catalog = "\n".join(
            f"[{i}] repo={s['_repo']} file={s['_file_path']} name={s['name']} type={s.get('type','')} "
            f"sig={s.get('signature','')[:100]} doc={(s.get('docstring') or '')[:100]}"
            for i, s in enumerate(candidates)
        )

        prompt = f"""You are a code search assistant. Given a query, rank the most relevant code symbols.

Query: {query}

Symbols:
{catalog}

Return ONLY a JSON array of indices of the top {top_k} most relevant symbols (e.g. [2, 0, 5]). No explanation."""

        # Build fallback chain: primary model first, then defaults (dedup preserving order)
        seen = set()
        model_chain = []
        for m in [MODEL] + _FALLBACK_MODELS:
            if m not in seen:
                seen.add(m)
                model_chain.append(m)

        indices = None
        for model in model_chain:
            try:
                response = ChatGPT_API(model, prompt, api_key=API_KEY)
                parsed = json.loads(response.strip())
                if isinstance(parsed, list):
                    indices = parsed
                    break
            except Exception:
                continue

        if indices is None:
            indices = list(range(min(top_k, len(candidates))))

        lines = [f"## Code search: {query}\n"]
        for idx in indices[:top_k]:
            if 0 <= idx < len(candidates):
                sym = candidates[idx]
                lr = sym.get("line_range", [0, 0])
                sym_key = f"{sym['_file_path']}:{sym['name']}"
                lines.append(f"### [{sym['_repo']}] {sym['_file_path']}:{sym['name']}")
                lines.append(f"**Type**: {sym.get('type', '')} | **Lines**: {lr[0]}-{lr[1]}")
                lines.append(f"```\n{sym.get('signature', '')}\n```")
                if sym.get("docstring"):
                    lines.append(f"> {sym['docstring'][:200]}")

                # Show callers from call graph
                callers_list = self._get_callers(sym_key, call_graph, depth=2)
                if callers_list:
                    lines.append(f"**Called by**: {', '.join(c[0] for c in callers_list[:5])}")
                lines.append("")

        return "\n".join(lines)

    def _prefilter(self, query: str, symbols: List[dict], max_candidates: int = 30) -> List[dict]:
        """Fast keyword-based pre-filter."""
        query_tokens = set(query.lower().split())
        # Add common code-related synonyms
        scored = []
        for sym in symbols:
            text = f"{sym.get('name', '')} {sym.get('docstring', '')} {sym.get('signature', '')} {sym.get('_file_path', '')}".lower()
            score = sum(1 for t in query_tokens if t in text)
            # Boost exact name matches
            if any(t == sym.get("name", "").lower() for t in query_tokens):
                score += 5
            if score > 0:
                scored.append((sym, score))

        scored.sort(key=lambda x: -x[1])
        return [s[0] for s in scored[:max_candidates]]
