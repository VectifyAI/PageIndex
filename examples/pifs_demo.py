"""
PageIndex FileSystem (PIFS) agent demo.

This mirrors examples/agentic_vectorless_rag_demo.py, but exposes a corpus
through the PageIndex FileSystem shell instead of direct PageIndex document
tools. The agent receives one read-only bash-like PIFS tool and must retrieve
evidence through commands such as ls, tree, find, grep, search-summary,
cat <path> --structure, cat <path> --page, and cat <path> --node.

The demo uses PDFs under examples/documents. When a matching
examples/documents/results/*_structure.json file exists, it is loaded into the
PIFS workspace's PageIndexClient cache so register() does not rebuild the tree.

Requirements:
  pip install openai-agents

Example:
  python examples/pifs_demo.py --stream-mode all --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import PyPDF2

sys.path.insert(0, str(Path(__file__).parent.parent))

# Keep the local demo quiet in offline environments.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "true")

from pageindex import PageIndexClient
from pageindex.filesystem import OpenAIMetadataGenerator, PageIndexFileSystem, PIFSCommandExecutor
from pageindex.filesystem.agent import run_pifs_agent


EXAMPLES_DIR = Path(__file__).parent
DOCUMENTS_DIR = EXAMPLES_DIR / "documents"
WORKSPACE = EXAMPLES_DIR / "pifs_workspace"
DEFAULT_MODEL = os.environ.get("PIFS_DEMO_MODEL", "gpt-5.4-mini")
DEFAULT_QUESTION = (
    "Use the PIFS workspace to find the Federal Reserve annual report. "
    "Which section covers supervision and regulation, and what page range "
    "should I inspect? Cite the document and evidence you used."
)

PIFS_DEMO_AGENT_PROMPT = """
You are a PageIndex FileSystem retrieval agent for a local demo workspace.

Use only the bash tool. It is a read-only PIFS virtual shell, not a real OS
shell. The workspace contains registered example PDFs.

Retrieval strategy:
- Start with ls or tree to understand the workspace.
- Use concrete PIFS paths from ls/find output, such as /documents/report.pdf,
  or stable file_ref/document ids. Do not invent temporary ref_N aliases.
- Folder paths such as /documents are positional command targets; do not put
  folder paths inside --where.
- Use search-summary when available to find likely documents.
  Quote multi-word queries and include a path, for example:
  search-summary "Federal Reserve supervision regulation" /documents
- Use find --where only with JSON metadata DSL, for example:
  find /documents --where '{"file_format":"pdf"}'
- Use grep -R only for lexical evidence; do not treat semantic candidates as
  literal matches.
- Run one evidence command at a time. Do not chain large commands like
  cat <path> --structure, grep, and cat <path> --page in one bash call.
- For PDFs, use cat <path> --structure to inspect the PageIndex tree, then
  cat <path> --page <range> for evidence, for example:
  cat /documents/2023-annual-report.pdf --page 31-35
- For page-range questions, use cat <path> --structure to identify the full section
  range. Then run cat <path> --page on the smallest useful evidence range, usually the
  section start page or first 1-2 pages, before the final answer. Do not print
  a broad multi-page section unless the user asks to read the whole section.
- Do not use cat --all on PDFs.
- Answer only from PIFS tool output and cite file refs or document ids.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PIFS document retrieval agent demo.")
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--documents-dir", type=Path, default=DOCUMENTS_DIR)
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        help="Specific document filename or path to register. May be repeated.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Limit number of cached example documents to register. 0 means all.",
    )
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the demo workspace.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Register documents and print PIFS smoke commands without running the agent.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--metadata-model",
        default=os.environ.get("PIFS_METADATA_MODEL", "gpt-5-nano"),
        help="OpenAI or OpenAI-compatible model used for register-time metadata.",
    )
    parser.add_argument("--stream-mode", default="all", choices=["off", "tools", "model", "all"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=90)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--reasoning-summary", default="auto")
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("PIFS_DEMO_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model used for register-time summary projection.",
    )
    parser.add_argument("--embedding-dimensions", type=int, default=256)
    return parser.parse_args()


def require_openai_environment() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "OPENAI_API_KEY is required for this demo: register() generates real "
        "PIFS metadata and the agent uses the OpenAI Agents SDK. Source your "
        ".env or export OPENAI_API_KEY before running."
    )


def discover_cached_documents(documents_dir: Path) -> list[Path]:
    results_dir = documents_dir / "results"
    paths: list[Path] = []
    for structure_path in sorted(results_dir.glob("*_structure.json")):
        stem = structure_path.name.removesuffix("_structure.json")
        for suffix in (".pdf", ".md", ".markdown"):
            candidate = documents_dir / f"{stem}{suffix}"
            if candidate.exists():
                paths.append(candidate)
                break
    return paths


def resolve_requested_documents(documents_dir: Path, requested: list[str]) -> list[Path]:
    if not requested:
        return discover_cached_documents(documents_dir)
    paths: list[Path] = []
    for item in requested:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = documents_dir / path
        if not path.exists():
            raise FileNotFoundError(f"document not found: {path}")
        paths.append(path)
    return paths


def structure_path_for(document_path: Path, documents_dir: Path) -> Path | None:
    path = documents_dir / "results" / f"{document_path.stem}_structure.json"
    return path if path.exists() else None


def deterministic_doc_id(document_path: Path) -> str:
    digest = hashlib.sha1(str(document_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"pifs_demo_{digest}"


def read_pdf_pages(document_path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with document_path.open("rb") as handle:
        reader = PyPDF2.PdfReader(handle)
        for page_num, page in enumerate(reader.pages, 1):
            pages.append({"page": page_num, "content": page.extract_text() or ""})
    return pages


def load_structure_json(structure_path: Path) -> dict[str, Any]:
    with structure_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("structure"), list):
        raise ValueError(f"invalid PageIndex structure cache: {structure_path}")
    return payload


def seed_pageindex_cache(
    filesystem: PageIndexFileSystem,
    document_path: Path,
    *,
    documents_dir: Path,
) -> str | None:
    structure_path = structure_path_for(document_path, documents_dir)
    if structure_path is None:
        return None

    filesystem.pageindex_client_workspace.mkdir(parents=True, exist_ok=True)
    meta_path = filesystem.pageindex_client_workspace / "_meta.json"
    if not meta_path.exists():
        meta_path.write_text("{}", encoding="utf-8")
    client = PageIndexClient(workspace=str(filesystem.pageindex_client_workspace))
    canonical_path = str(document_path.resolve())
    for doc_id, doc in client.documents.items():
        if Path(str(doc.get("path") or "")).resolve(strict=False) == Path(canonical_path):
            return doc_id

    payload = load_structure_json(structure_path)
    doc_id = deterministic_doc_id(document_path)
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        pages = read_pdf_pages(document_path)
        client.documents[doc_id] = {
            "id": doc_id,
            "type": "pdf",
            "path": canonical_path,
            "doc_name": payload.get("doc_name") or document_path.name,
            "doc_description": payload.get("doc_description") or "",
            "page_count": len(pages),
            "structure": payload["structure"],
            "pages": pages,
        }
    elif suffix in {".md", ".markdown"}:
        text = document_path.read_text(encoding="utf-8")
        client.documents[doc_id] = {
            "id": doc_id,
            "type": "md",
            "path": canonical_path,
            "doc_name": payload.get("doc_name") or document_path.name,
            "doc_description": payload.get("doc_description") or "",
            "line_count": len(text.splitlines()),
            "structure": payload["structure"],
        }
    else:
        return None
    client._save_doc(doc_id)
    return doc_id


def content_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".md", ".markdown"}:
        return "text/markdown"
    return "text/plain"


def external_id_for(path: Path) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in path.stem).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)
    return f"example_{slug}"


def log_progress(message: str, *, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}{message}", flush=True)


def register_demo_metadata_schema(filesystem: PageIndexFileSystem) -> None:
    filesystem.metadata.register_schema(
        {
            "fields": {
                "source_collection": {
                    "type": "string",
                    "description": "Local example corpus collection.",
                },
                "file_format": {
                    "type": "string",
                    "description": "Source file extension without the leading dot.",
                },
            }
        },
        source="demo",
    )


def backfill_registered_metadata_values(filesystem: PageIndexFileSystem, file_ref: str) -> None:
    entry = filesystem.store.get_file(file_ref)
    indexed_metadata = dict(entry.metadata or {})
    with filesystem.store.connect() as conn:
        filesystem.store.replace_metadata_values(conn, file_ref, indexed_metadata)


def configure_summary_projection_backend(
    filesystem: PageIndexFileSystem,
    *,
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    if not (filesystem.summary_projection_index_dir / "summary_only_vector.sqlite").exists():
        return
    filesystem.configure_hybrid_projection_retrieval(
        filesystem.summary_projection_index_dir,
        embedding_provider="openai",
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )


def has_ready_register_outputs(filesystem: PageIndexFileSystem, external_id: str) -> bool:
    try:
        file_ref = filesystem.store.resolve_file_ref(external_id)
        entry = filesystem.store.get_file(file_ref)
    except KeyError:
        return False
    status = entry.metadata_status or {}
    fields = status.get("fields") or {}
    required = ("summary", "doc_type", "domain", "topic")
    if any(fields.get(field, {}).get("status") != "generated" for field in required):
        return False
    summary_projection = (status.get("projection_indexes") or {}).get("summary") or {}
    return summary_projection.get("status") == "ready"


def register_documents(
    filesystem: PageIndexFileSystem,
    documents: list[Path],
    *,
    documents_dir: Path,
) -> list[dict[str, Any]]:
    registered: list[dict[str, Any]] = []
    total = len(documents)
    for index, document_path in enumerate(documents, 1):
        document_path = document_path.resolve()
        external_id = external_id_for(document_path)
        log_progress(f"[{index}/{total}] {document_path.name}")
        log_progress("PageIndex tree cache: checking examples/documents/results", indent=1)
        cache_started = time.perf_counter()
        cached_doc_id = seed_pageindex_cache(
            filesystem,
            document_path,
            documents_dir=documents_dir,
        )
        cache_seconds = time.perf_counter() - cache_started
        if cached_doc_id:
            log_progress(
                f"PageIndex tree cache: ready doc_id={cached_doc_id} ({cache_seconds:.2f}s)",
                indent=1,
            )
        else:
            log_progress(
                f"PageIndex tree cache: no cached structure; register() will index if supported ({cache_seconds:.2f}s)",
                indent=1,
            )
        if has_ready_register_outputs(filesystem, external_id):
            file_ref = filesystem.store.resolve_file_ref(external_id)
            backfill_registered_metadata_values(filesystem, file_ref)
            log_progress(
                f"PIFS register: cached file_ref={file_ref}; metadata and summary projection already ready",
                indent=1,
            )
            registered.append(
                {
                    "file_ref": file_ref,
                    "external_id": external_id,
                    "path": str(document_path),
                    "status": "cached",
                    "pageindex_doc_id": cached_doc_id,
                }
            )
            continue

        log_progress(
            "PIFS register: running register() -> metadata generation -> summary embedding -> sqlite upsert",
            indent=1,
        )
        register_started = time.perf_counter()
        file_ref = filesystem.register(
            storage_uri=document_path.as_uri(),
            source_path=str(document_path),
            folder_path="/documents",
            external_id=external_id,
            title=document_path.name,
            content_type=content_type_for(document_path),
            source_type="examples-documents",
            metadata={
                "title": document_path.name,
                "source_collection": "examples/documents",
                "file_format": document_path.suffix.lower().lstrip("."),
            },
        )
        register_seconds = time.perf_counter() - register_started
        entry = filesystem.store.get_file(file_ref)
        field_status = {
            field: state.get("status")
            for field, state in (entry.metadata_status.get("fields") or {}).items()
        }
        summary_projection = (
            entry.metadata_status.get("projection_indexes", {}).get("summary", {})
        )
        log_progress(
            f"PIFS register: done file_ref={file_ref} ({register_seconds:.2f}s)",
            indent=1,
        )
        log_progress(
            f"metadata: {entry.metadata_status.get('status', 'unknown')} fields={field_status}",
            indent=1,
        )
        log_progress(
            "summary projection: "
            f"{summary_projection.get('status', 'not_requested')} "
            f"index={summary_projection.get('index_path', '')}",
            indent=1,
        )
        registered.append(
            {
                "file_ref": file_ref,
                "external_id": external_id,
                "path": str(document_path),
                "status": entry.metadata_status.get("status", "unknown"),
                "pageindex_tree_status": entry.pageindex_tree_status,
                "pageindex_doc_id": entry.pageindex_doc_id,
            }
        )
    return registered


def print_section(title: str) -> None:
    print("\n" + "#" * 78, flush=True)
    print(f"# {title}", flush=True)
    print("#" * 78, flush=True)


def print_step(title: str, detail: str = "") -> None:
    print(f"\n>>> {title}", flush=True)
    if detail:
        print(f"    {detail}", flush=True)


def sanitize_preview_text(text: str) -> str:
    cleaned = str(text).replace("\r", "\n").replace("\f", "\n")
    cleaned = "".join(
        ch if ch == "\n" or ch == "\t" or ord(ch) >= 32 else " "
        for ch in cleaned
    )
    return "\n".join(
        re.sub(r"[ \t]{2,}", " ", line).strip()
        for line in cleaned.splitlines()
    )


def compact_lines(text: str, *, max_lines: int = 6, max_chars: int = 900) -> str:
    lines = [line for line in sanitize_preview_text(text).splitlines() if line.strip()]
    preview = "\n".join(lines[:max_lines])
    if len(preview) > max_chars:
        preview = preview[:max_chars].rstrip() + "..."
    omitted = len(lines) - min(len(lines), max_lines)
    if omitted > 0:
        preview += f"\n    ... {omitted} more lines"
    return preview


def find_structure_node(structure: Any, title_fragment: str) -> dict[str, Any] | None:
    if isinstance(structure, list):
        for item in structure:
            found = find_structure_node(item, title_fragment)
            if found:
                return found
        return None
    if not isinstance(structure, dict):
        return None
    if title_fragment.lower() in str(structure.get("title", "")).lower():
        return structure
    return find_structure_node(structure.get("nodes", []), title_fragment)


def page_range_for_node(node: dict[str, Any] | None) -> str:
    if not node:
        return ""
    ranges: list[tuple[int, int]] = []

    def collect(item: Any) -> None:
        if not isinstance(item, dict):
            return
        start = item.get("start_index")
        end = item.get("end_index")
        if isinstance(start, int) and isinstance(end, int):
            ranges.append((start, end))
        for child in item.get("nodes") or []:
            collect(child)

    collect(node)
    if not ranges:
        return ""
    start = min(item[0] for item in ranges)
    end = max(item[1] for item in ranges)
    return f"{start}-{end}" if start != end else str(start)


def opening_page_range_for_node(node: dict[str, Any] | None, *, max_pages: int = 2) -> str:
    if not node:
        return ""
    ranges: list[tuple[int, int]] = []

    def collect(item: Any) -> None:
        if not isinstance(item, dict):
            return
        start = item.get("start_index")
        end = item.get("end_index")
        if isinstance(start, int) and isinstance(end, int):
            ranges.append((start, end))
        for child in item.get("nodes") or []:
            collect(child)

    collect(node)
    if not ranges:
        return ""
    start = min(item[0] for item in ranges)
    end = max(item[1] for item in ranges)
    preview_end = min(end, start + max_pages - 1)
    return f"{start}-{preview_end}" if start != preview_end else str(start)


def execute_json_command(executor: PIFSCommandExecutor, command: str) -> dict[str, Any]:
    try:
        return json.loads(executor.execute(command))
    except Exception as exc:
        return {"ok": False, "error": str(exc), "data": None}


def show_capability(
    *,
    label: str,
    command: str,
    result: str,
    raw: str = "",
    verbose: bool = False,
) -> None:
    print_step(label, command)
    print(f"    result: {result}", flush=True)
    if verbose and raw:
        print("    raw:", flush=True)
        print(compact_lines(raw, max_lines=10, max_chars=1600), flush=True)


def show_registered_documents(registered: list[dict[str, Any]], *, verbose: bool = False) -> None:
    print(f"\nRegistered {len(registered)} document(s):", flush=True)
    for item in registered:
        print(
            "  - "
            f"{Path(str(item.get('path', ''))).name}: "
            f"file_ref={item.get('file_ref')} | "
            f"status={item.get('status')} | "
            f"pageindex_doc_id={item.get('pageindex_doc_id')}",
            flush=True,
        )
    if verbose:
        print("\nRaw registration records:", flush=True)
        print(json.dumps(registered, ensure_ascii=False, indent=2), flush=True)


def run_smoke_commands(
    filesystem: PageIndexFileSystem,
    registered: list[dict[str, Any]],
    *,
    verbose: bool = False,
) -> None:
    json_executor = PIFSCommandExecutor(filesystem, json_output=True)
    shell_executor = PIFSCommandExecutor(filesystem, json_output=False)

    command = "tree / --depth 2"
    tree = execute_json_command(json_executor, command)
    folders = (tree.get("data") or {}).get("folders") or []
    documents_folder = next((item for item in folders if item.get("path") == "/documents"), {})
    show_capability(
        label="Folder browse",
        command=command,
        result=f"/documents contains {documents_folder.get('file_count', len(registered))} files",
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    command = "ls /documents"
    listing = execute_json_command(json_executor, command)
    files = (listing.get("data") or {}).get("files") or []
    file_titles = ", ".join(item.get("title", "") for item in files[:3])
    show_capability(
        label="List registered files",
        command=command,
        result=f"{len(files)} files: {file_titles}",
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    command = "stat --schema"
    schema = execute_json_command(json_executor, command)
    fields = sorted(((schema.get("data") or {}).get("fields") or {}).keys())
    show_capability(
        label="Metadata schema",
        command=command,
        result=", ".join(fields),
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    command = "find /documents --where '{\"source_collection\":\"examples/documents\"}' --limit 5"
    found = execute_json_command(json_executor, command)
    found_files = found.get("data") or []
    show_capability(
        label="Metadata DSL filter",
        command=command,
        result=f"{len(found_files)} documents matched source_collection=examples/documents",
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    command = 'search-summary "Federal Reserve annual report supervision regulation section page range" /documents'
    summary = execute_json_command(json_executor, command)
    summary_hits = ((summary.get("data") or {}).get("data") or [])
    if summary_hits:
        summary_result = f"{len(summary_hits)} summary-vector candidates; top={summary_hits[0].get('external_id')}"
    else:
        summary_result = "summary-vector command is available, but this tiny two-doc demo returned no candidates"
    show_capability(
        label="Semantic summary search",
        command=command,
        result=summary_result,
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    first_target = f"/documents/{Path(str(registered[0]['path'])).name}" if registered else None
    if not first_target:
        return

    command = f"stat {first_target}"
    stat = execute_json_command(json_executor, command)
    stat_data = stat.get("data") or {}
    show_capability(
        label="File stat",
        command=command,
        result=(
            f"{stat_data.get('title')} | tree={stat_data.get('pageindex_tree_status')} | "
            f"metadata_status={(stat_data.get('metadata_status') or {}).get('status')}"
        ),
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    command = f"cat {first_target} --structure"
    structure_payload = execute_json_command(json_executor, command)
    structure_data = structure_payload.get("data") or {}
    structure = structure_data.get("structure") or []
    supervision_node = find_structure_node(structure, "Supervision and Regulation")
    supervision_range = page_range_for_node(supervision_node)
    show_capability(
        label="PageIndex document structure",
        command=command,
        result=(
            "found section 'Supervision and Regulation'"
            + (f" with page span {supervision_range}" if supervision_range else "")
        ),
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    evidence_range = opening_page_range_for_node(supervision_node) or "1-2"
    command = f"cat {first_target} --page {evidence_range}"
    page = execute_json_command(json_executor, command)
    page_text = str((page.get("data") or {}).get("text") or "")
    show_capability(
        label="Page evidence",
        command=command,
        result=compact_lines(page_text, max_lines=3, max_chars=420),
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )

    command = 'grep -R "Supervision and Regulation" /documents'
    grep = execute_json_command(json_executor, command)
    grep_hits = ((grep.get("data") or {}).get("data") or [])
    show_capability(
        label="Lexical grep",
        command=command,
        result=f"{len(grep_hits)} real text matches",
        raw=shell_executor.execute(command) if verbose else "",
        verbose=verbose,
    )


def main() -> None:
    args = parse_args()
    require_openai_environment()
    workspace = args.workspace.expanduser()
    documents_dir = args.documents_dir.expanduser()
    if args.reset and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    documents = resolve_requested_documents(documents_dir, args.document)
    if args.max_docs > 0:
        documents = documents[: args.max_docs]
    if not documents:
        raise RuntimeError(f"no cached example documents found under {documents_dir}")

    filesystem = PageIndexFileSystem(
        workspace,
        metadata_generator=OpenAIMetadataGenerator(model=args.metadata_model),
        summary_projection_embedding_provider="openai",
        summary_projection_embedding_model=args.embedding_model,
        summary_projection_embedding_dimensions=args.embedding_dimensions,
    )
    register_demo_metadata_schema(filesystem)

    print_section("STEP 1/3  Register Documents")
    print(f"Workspace: {workspace}", flush=True)
    print(f"Documents: {len(documents)}", flush=True)
    registered = register_documents(filesystem, documents, documents_dir=documents_dir)
    configure_summary_projection_backend(
        filesystem,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.embedding_dimensions,
    )
    show_registered_documents(registered, verbose=args.verbose)

    print_section("STEP 2/3  Explore PIFS Tool Surface")
    run_smoke_commands(filesystem, registered, verbose=args.verbose)

    if args.prepare_only:
        return

    print_section("STEP 3/3  Ask An Agent Using Only PIFS")
    print(f"Question: {args.question}", flush=True)
    answer = run_pifs_agent(
        filesystem,
        args.question,
        model=args.model,
        root="/",
        system_prompt=PIFS_DEMO_AGENT_PROMPT,
        max_turns=args.max_turns,
        max_seconds=args.max_seconds,
        verbose=args.verbose,
        stream_mode=args.stream_mode,
        reasoning_effort=args.reasoning_effort,
        reasoning_summary=args.reasoning_summary,
    )
    if answer:
        print("\nFinal answer:", flush=True)
        print(answer, flush=True)


if __name__ == "__main__":
    main()
