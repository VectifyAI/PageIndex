"""
batch_processor.py — Multi-document processing and knowledge base search.

Public API
----------
process_batch(doc_paths, output_dir, **kwargs) -> dict
    Process a list of PDF files and write per-document structure JSONs plus a
    kb_index.json manifest to output_dir.

KnowledgeBaseSearch(kb_index_path)
    Load a knowledge base produced by process_batch() and search across it.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

from pageindex.page_index import page_index_main
from pageindex.utils import ConfigLoader, config as Config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_nodes(node: dict, doc_name: str, out: list) -> None:
    """Recursively collect every node from a structure tree into *out*."""
    entry = {
        "doc_name": doc_name,
        "title": node.get("title", ""),
        "start_index": node.get("start_index"),
        "end_index": node.get("end_index"),
    }
    if "node_id" in node:
        entry["node_id"] = node["node_id"]
    if "summary" in node:
        entry["summary"] = node["summary"]
    out.append(entry)
    for child in node.get("nodes", []):
        _flatten_nodes(child, doc_name, out)


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------

def process_batch(
    doc_paths: List[str],
    output_dir: str = "./results",
    model: Optional[str] = None,
    toc_check_page_num: Optional[int] = None,
    max_page_num_each_node: Optional[int] = None,
    max_token_num_each_node: Optional[int] = None,
    if_add_node_id: Optional[str] = None,
    if_add_node_summary: Optional[str] = None,
    if_add_doc_description: Optional[str] = None,
    if_add_node_text: Optional[str] = None,
    batch_logger: Optional[logging.Logger] = None,
) -> dict:
    """Process multiple PDF files and build a knowledge-base index.

    For each PDF, the document's hierarchical structure is saved as::

        {output_dir}/{stem}_structure.json

    A manifest file is also written::

        {output_dir}/kb_index.json

    The manifest lists every document, its status, and its output file so that
    :class:`KnowledgeBaseSearch` can load and query the results.

    Args:
        doc_paths: List of PDF file paths to process.
        output_dir: Directory for output files (created if absent).
        model: LLM model name. Falls back to ``config.yaml`` default when ``None``.
        toc_check_page_num: Pages to scan for a table of contents.
        max_page_num_each_node: Maximum pages per tree node.
        max_token_num_each_node: Maximum tokens per tree node.
        if_add_node_id: ``'yes'``/``'no'`` — attach node IDs.
        if_add_node_summary: ``'yes'``/``'no'`` — generate node summaries.
        if_add_doc_description: ``'yes'``/``'no'`` — generate a document description.
        if_add_node_text: ``'yes'``/``'no'`` — include raw page text per node.
        batch_logger: Optional logger; falls back to the module-level logger.

    Returns:
        A dict with keys:

        - ``"processed"`` — list of filenames that succeeded.
        - ``"failed"``    — list of ``{"doc": path, "error": message}`` dicts.
        - ``"output_dir"``   — absolute path to the output directory.
        - ``"kb_index_path"`` — absolute path to ``kb_index.json``.

    Raises:
        ValueError: if *doc_paths* is empty or contains non-PDF files.
    """
    log = batch_logger or logger

    if not doc_paths:
        raise ValueError("doc_paths must not be empty.")

    non_pdf = [p for p in doc_paths if not p.lower().endswith(".pdf")]
    if non_pdf:
        raise ValueError(
            f"Batch processing only supports PDF files. "
            f"Unsupported files: {non_pdf}"
        )

    os.makedirs(output_dir, exist_ok=True)
    abs_output_dir = os.path.abspath(output_dir)

    # Build shared opt — skip keys whose value is None so defaults apply.
    user_opt = {
        k: v for k, v in {
            "model": model,
            "toc_check_page_num": toc_check_page_num,
            "max_page_num_each_node": max_page_num_each_node,
            "max_token_num_each_node": max_token_num_each_node,
            "if_add_node_id": if_add_node_id,
            "if_add_node_summary": if_add_node_summary,
            "if_add_doc_description": if_add_doc_description,
            "if_add_node_text": if_add_node_text,
        }.items() if v is not None
    }
    opt = ConfigLoader().load(user_opt)

    processed: List[str] = []
    failed: List[dict] = []
    kb_documents: List[dict] = []

    for doc_path in doc_paths:
        doc_name = os.path.basename(doc_path)
        stem = os.path.splitext(doc_name)[0]
        output_file = os.path.join(abs_output_dir, f"{stem}_structure.json")

        if not os.path.isfile(doc_path):
            failed.append({"doc": doc_path, "error": "File not found."})
            log.warning(f"Skipping '{doc_name}': file not found.")
            continue

        try:
            log.info(f"Processing '{doc_name}' ...")
            result = page_index_main(doc_path, opt)

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            kb_documents.append({
                "doc_name": result.get("doc_name", doc_name),
                "doc_description": result.get("doc_description", ""),
                "output_file": os.path.basename(output_file),
                "status": "success",
            })
            processed.append(doc_name)
            log.info(f"Saved '{output_file}'")

        except Exception as exc:
            failed.append({"doc": doc_path, "error": str(exc)})
            log.error(f"Failed to process '{doc_name}': {exc}")

    kb_index = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_documents": len(kb_documents),
        "documents": kb_documents,
    }
    kb_index_path = os.path.join(abs_output_dir, "kb_index.json")
    with open(kb_index_path, "w", encoding="utf-8") as f:
        json.dump(kb_index, f, indent=2, ensure_ascii=False)

    log.info(
        f"Batch complete — processed: {len(processed)}, "
        f"failed: {len(failed)}. Index: {kb_index_path}"
    )
    return {
        "processed": processed,
        "failed": failed,
        "output_dir": abs_output_dir,
        "kb_index_path": kb_index_path,
    }


# ---------------------------------------------------------------------------
# KnowledgeBaseSearch
# ---------------------------------------------------------------------------

class KnowledgeBaseSearch:
    """Load a knowledge base produced by :func:`process_batch` and search it.

    Searching is case-insensitive substring matching across node titles and
    summaries — no embeddings needed, consistent with PageIndex's
    reasoning-first philosophy.

    Scoring: a title match scores 2, a summary match scores 1.  Results are
    returned in descending score order.

    Example::

        kb = KnowledgeBaseSearch("./results/kb_index.json")
        hits = kb.search("revenue growth", top_k=5)
        full = kb.get_document("earnings_report.pdf")
        names = kb.list_documents()
    """

    def __init__(self, kb_index_path: str) -> None:
        """
        Args:
            kb_index_path: Path to ``kb_index.json`` produced by
                :func:`process_batch`.

        Raises:
            FileNotFoundError: if the index file does not exist.
        """
        if not os.path.isfile(kb_index_path):
            raise FileNotFoundError(
                f"Knowledge base index not found: {kb_index_path}"
            )
        with open(kb_index_path, "r", encoding="utf-8") as f:
            self._index: dict = json.load(f)

        self._results_dir = os.path.dirname(os.path.abspath(kb_index_path))
        self._structures: dict = {}          # doc_name -> loaded structure dict
        self._flat_nodes: Optional[list] = None  # flattened once, then cached

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_structure(self, doc_entry: dict) -> dict:
        doc_name = doc_entry["doc_name"]
        if doc_name not in self._structures:
            path = os.path.join(self._results_dir, doc_entry["output_file"])
            with open(path, "r", encoding="utf-8") as f:
                self._structures[doc_name] = json.load(f)
        return self._structures[doc_name]

    def _get_all_nodes(self) -> list:
        if self._flat_nodes is not None:
            return self._flat_nodes

        self._flat_nodes = []
        for doc_entry in self._index.get("documents", []):
            if doc_entry.get("status") != "success":
                continue
            try:
                structure = self._load_structure(doc_entry)
                doc_name = structure.get("doc_name", doc_entry["doc_name"])
                for top_node in structure.get("structure", []):
                    _flatten_nodes(top_node, doc_name, self._flat_nodes)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
        return self._flat_nodes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list:
        """Search across all document nodes by title and summary.

        Args:
            query: Case-insensitive search string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts sorted by score (highest first)::

                {
                    "doc_name":    str,
                    "title":       str,
                    "start_index": int,
                    "end_index":   int,
                    "score":       int,   # 1–3
                    "match_on":    list,  # ["title"] / ["summary"] / both
                    "node_id":     str,   # if present
                    "summary":     str,   # if present
                }

        Raises:
            ValueError: if *query* is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        q = query.lower()
        scored = []

        for node in self._get_all_nodes():
            score = 0
            match_on = []

            if q in node["title"].lower():
                score += 2
                match_on.append("title")
            if q in node.get("summary", "").lower():
                score += 1
                match_on.append("summary")

            if score > 0:
                result = {
                    "doc_name": node["doc_name"],
                    "title": node["title"],
                    "start_index": node["start_index"],
                    "end_index": node["end_index"],
                    "score": score,
                    "match_on": match_on,
                }
                if "node_id" in node:
                    result["node_id"] = node["node_id"]
                if "summary" in node:
                    result["summary"] = node["summary"]
                scored.append(result)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_document(self, doc_name: str) -> dict:
        """Retrieve the full structure dict for a specific document.

        Args:
            doc_name: The ``doc_name`` as stored in the index
                (e.g. ``"earnings_report.pdf"``).

        Returns:
            The full structure dict: ``{doc_name, structure, doc_description}``.

        Raises:
            KeyError: if *doc_name* is not found in the knowledge base.
        """
        for doc_entry in self._index.get("documents", []):
            if doc_entry["doc_name"] == doc_name:
                return self._load_structure(doc_entry)
        raise KeyError(f"Document '{doc_name}' not found in the knowledge base.")

    def list_documents(self) -> List[str]:
        """Return the names of all successfully processed documents.

        Returns:
            List of ``doc_name`` strings.
        """
        return [
            d["doc_name"]
            for d in self._index.get("documents", [])
            if d.get("status") == "success"
        ]
