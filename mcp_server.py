#!/usr/bin/env python3
"""
PageIndex MCP Server
Exposes: index_document, search, list_documents
LLM: MiniMax via OPENAI_BASE_URL env var
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Add pageindex package to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from pageindex.page_index_md import md_to_tree
from pageindex.utils import ChatGPT_API, structure_to_list
from pageindex.code_indexer import CodeIndexer
from pageindex.code_searcher import CodeSearcher

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = os.getenv("PAGEINDEX_MODEL", "MiniMax-M2.7-highspeed")
API_KEY = os.getenv("CHATGPT_API_KEY")

server = Server("pageindex")
_code_indexer = CodeIndexer()
_code_searcher = CodeSearcher()


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="index_document",
            description="Index a Markdown file with PageIndex tree structure. Use before searching a new document.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the .md file"},
                    "doc_name": {"type": "string", "description": "Short name for this document (used for retrieval)"},
                },
                "required": ["path"],
            },
        ),
        Tool(
            name="search",
            description=(
                "Search across PageIndex-indexed documents using LLM tree navigation. "
                "More precise than vector search for structured documents. "
                "Use for: 'what projects do we have', 'infra details', 'trading system config', etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language question"},
                    "doc_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Limit to specific documents (omit to search all)",
                    },
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_documents",
            description="List all documents currently indexed by PageIndex.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="index_code",
            description="Index a code repository using tree-sitter AST parsing. Extracts functions, classes, constants, imports, and call graphs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Absolute path to the code repository root"},
                    "language_filter": {"type": "string", "description": "Optional: only index specific language (py, ts, python, typescript)"},
                    "repo_name": {"type": "string", "description": "Optional: custom name for the index (defaults to directory name)"},
                    "max_files": {"type": "integer", "default": 500, "description": "Max files to index (safety limit)"},
                },
                "required": ["dir_path"],
            },
        ),
        Tool(
            name="search_code",
            description="Search indexed code repositories. Modes: 'relevance' (LLM-ranked semantic search), 'definition' (find symbol by name), 'impact' (find callers of a symbol).",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language query or symbol name"},
                    "repo_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Limit to specific repos (omit to search all)",
                    },
                    "top_k": {"type": "integer", "default": 5},
                    "mode": {"type": "string", "enum": ["relevance", "definition", "impact"], "default": "relevance"},
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "index_document":
        return await _index_document(arguments)
    elif name == "search":
        return await _search(arguments)
    elif name == "list_documents":
        return _list_documents()
    elif name == "index_code":
        return await _index_code(arguments)
    elif name == "search_code":
        return await _search_code(arguments)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _index_document(args: dict) -> list[TextContent]:
    path = args["path"]
    if not os.path.isfile(path):
        return [TextContent(type="text", text=f"File not found: {path}")]

    doc_name = args.get("doc_name") or Path(path).stem
    out_file = RESULTS_DIR / f"{doc_name}_structure.json"

    try:
        tree = await md_to_tree(
            md_path=path,
            if_add_node_summary="yes",
            if_add_node_id="yes",
            model=MODEL,
        )
        tree["doc_name"] = doc_name
        tree["source_path"] = path

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)

        node_count = len(structure_to_list(tree.get("structure", [])))
        return [TextContent(type="text", text=f"Indexed '{doc_name}': {node_count} nodes → {out_file.name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Indexing error: {e}")]


def _list_documents() -> list[TextContent]:
    lines = []

    # Markdown indices
    md_files = sorted(RESULTS_DIR.glob("*_structure.json"))
    if md_files:
        lines.append("## Markdown documents")
        for f in md_files:
            try:
                with open(f, encoding="utf-8") as fp:
                    d = json.load(fp)
                nodes = len(structure_to_list(d.get("structure", [])))
                src = d.get("source_path", "")
                lines.append(f"- **{d.get('doc_name', f.stem)}** ({nodes} nodes) ← {src}")
            except Exception:
                lines.append(f"- {f.stem} (unreadable)")

    # Code indices
    code_files = sorted(RESULTS_DIR.glob("*_code_index.json"))
    if code_files:
        lines.append("\n## Code repositories")
        for f in code_files:
            try:
                with open(f, encoding="utf-8") as fp:
                    d = json.load(fp)
                lines.append(
                    f"- **{d.get('repo_name', f.stem)}** "
                    f"({d.get('total_files', 0)} files, {d.get('total_symbols', 0)} symbols) "
                    f"← {d.get('repo_path', '')}"
                )
            except Exception:
                lines.append(f"- {f.stem} (unreadable)")

    if not lines:
        return [TextContent(type="text", text="No documents indexed yet. Use index_document or index_code first.")]
    return [TextContent(type="text", text="\n".join(lines))]


async def _search(args: dict) -> list[TextContent]:
    query = args["query"]
    filter_names = set(args.get("doc_names") or [])
    top_k = args.get("top_k", 3)

    # Load all (or filtered) indexed trees
    files = sorted(RESULTS_DIR.glob("*_structure.json"))
    all_nodes = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fp:
                d = json.load(fp)
            doc_name = d.get("doc_name", f.stem)
            if filter_names and doc_name not in filter_names:
                continue
            for node in structure_to_list(d.get("structure", [])):
                if node.get("title") and node.get("title") != d.get("doc_name"):
                    all_nodes.append({
                        "doc": doc_name,
                        "title": node.get("title", ""),
                        "node_id": node.get("node_id", ""),
                        "summary": node.get("summary", node.get("prefix_summary", "")),
                    })
        except Exception:
            continue

    if not all_nodes:
        return [TextContent(type="text", text="No indexed documents found. Run index_document first.")]

    # Build node catalog for LLM to rank
    catalog = "\n".join(
        f"[{i}] doc={n['doc']} | id={n['node_id']} | title={n['title']} | summary={n['summary'][:200]}"
        for i, n in enumerate(all_nodes)
    )

    prompt = f"""You are a memory retrieval assistant. Given a user query, identify the {top_k} most relevant nodes from the document index below.

Query: {query}

Document nodes:
{catalog}

Return ONLY a JSON array of the indices (e.g. [2, 7, 14]) of the top {top_k} most relevant nodes. No explanation."""

    try:
        response = ChatGPT_API(MODEL, prompt, api_key=API_KEY)
        # Parse indices
        indices = json.loads(response.strip())
        if not isinstance(indices, list):
            raise ValueError("not a list")
    except Exception:
        # Fallback: return first top_k nodes
        indices = list(range(min(top_k, len(all_nodes))))

    lines = [f"## PageIndex search: {query}\n"]
    for idx in indices[:top_k]:
        if 0 <= idx < len(all_nodes):
            n = all_nodes[idx]
            lines.append(f"### [{n['doc']}] {n['title']}")
            lines.append(n["summary"] or "(no summary)")
            lines.append("")

    return [TextContent(type="text", text="\n".join(lines))]


async def _index_code(args: dict) -> list[TextContent]:
    dir_path = args["dir_path"]
    if not os.path.isdir(dir_path):
        return [TextContent(type="text", text=f"Directory not found: {dir_path}")]

    max_files = args.get("max_files", 500)
    indexer = CodeIndexer(max_files=max_files)
    try:
        index = await indexer.index_directory(
            dir_path=dir_path,
            language_filter=args.get("language_filter"),
            repo_name=args.get("repo_name"),
        )
        out_file = RESULTS_DIR / f"{index.repo_name}_code_index.json"
        indexer.save_index(index, str(out_file))
        return [TextContent(
            type="text",
            text=f"Indexed '{index.repo_name}': {index.total_files} files, {index.total_symbols} symbols → {out_file.name}",
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Code indexing error: {e}")]


async def _search_code(args: dict) -> list[TextContent]:
    query = args["query"]
    try:
        result = await _code_searcher.search(
            query=query,
            repo_names=args.get("repo_names"),
            top_k=args.get("top_k", 5),
            mode=args.get("mode", "relevance"),
        )
        return [TextContent(type="text", text=result)]
    except Exception as e:
        return [TextContent(type="text", text=f"Code search error: {e}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
