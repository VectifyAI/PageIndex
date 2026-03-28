"""
run_pageindex.py — CLI for PageIndex document tree generation.

Usage examples:
  python run_pageindex.py --pdf_path report.pdf
  python run_pageindex.py --md_path notes.md --output-format markdown
  python run_pageindex.py --pdf_path paper.pdf --model anthropic/claude-sonnet-4-6 --verbose
  python run_pageindex.py --pdf_path paper.pdf --output-dir ./my_results
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Optional

from pageindex import page_index_main
from pageindex.page_index_md import md_to_tree
from pageindex.utils import ConfigLoader

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_key(model: Optional[str]) -> None:
    """Verify that the required API key is present for the chosen model.

    Detects the provider from the LiteLLM model string prefix and checks
    the corresponding environment variable.  Exits with a helpful message
    if the key is missing.

    Args:
        model: LiteLLM model string, e.g. ``"gpt-4o-2024-11-20"``,
               ``"anthropic/claude-sonnet-4-6"``, ``"gemini/gemini-2.0-flash"``.
               ``None`` falls back to the OpenAI check.
    """
    m = (model or "").lower()

    if m.startswith("anthropic/") or m.startswith("claude"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            sys.exit(
                "Error: ANTHROPIC_API_KEY is not set.\n"
                "  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  Or add it to a .env file in the project root.\n"
                "  Get a key:    https://console.anthropic.com/settings/keys"
            )
    elif m.startswith("gemini/"):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            sys.exit(
                "Error: GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.\n"
                "  Set it with:  export GEMINI_API_KEY=...\n"
                "  Or add it to a .env file in the project root.\n"
                "  Get a key:    https://ai.google.dev/gemini-api/docs/api-key"
            )
    else:
        # Default provider: OpenAI (also accepts CHATGPT_API_KEY alias)
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")):
            sys.exit(
                "Error: OPENAI_API_KEY is not set.\n"
                "  Set it with:  export OPENAI_API_KEY=sk-...\n"
                "  Or add it to a .env file in the project root.\n"
                "  Get a key:    https://platform.openai.com/api-keys\n"
                "\n"
                "  Using a different provider? Pass --model with a prefix:\n"
                "    --model anthropic/claude-sonnet-4-6\n"
                "    --model gemini/gemini-2.0-flash"
            )


def _render_node_md(node: dict, depth: int, lines: list) -> None:
    """Recursively render one tree node as Markdown headings.

    Args:
        node:  Tree node dict (may contain title, summary, start_index,
               end_index, line_num, nodes).
        depth: Current heading depth — 0 produces ``##``, 1 produces ``###``,
               capped at ``######``.
        lines: Accumulator list that receives rendered Markdown lines.
    """
    heading = "#" * min(depth + 2, 6)
    title = node.get("title", "Untitled")

    # Location hint: page range for PDF, line number for Markdown
    start = node.get("start_index") or node.get("line_num") or ""
    end = node.get("end_index", "")
    if start and end:
        loc = f" *(pages {start}–{end})*"
    elif start:
        loc = f" *(line {start})*"
    else:
        loc = ""

    lines.append(f"{heading} {title}{loc}")
    if node.get("summary"):
        lines.append(f"\n{node['summary']}")
    lines.append("")

    for child in node.get("nodes") or []:
        _render_node_md(child, depth + 1, lines)


def structure_to_markdown(result: dict) -> str:
    """Convert a PageIndex tree structure dict to a Markdown document outline.

    Produces a readable outline with headings, page/line location hints, and
    LLM-generated summaries (when present in the structure).

    Args:
        result: The dict returned by :func:`page_index_main` or
                :func:`md_to_tree`.  Expected top-level keys:
                ``doc_name``, ``doc_description`` (optional),
                ``structure`` (list of tree nodes).

    Returns:
        A Markdown string representing the full document tree.
    """
    lines: list = []

    doc_name = result.get("doc_name", "Document")
    lines.append(f"# {doc_name}")
    lines.append("")

    if result.get("doc_description"):
        lines.append(f"> {result['doc_description']}")
        lines.append("")

    structure = result.get("structure", [])
    top_nodes = (
        _tqdm(structure, desc="Rendering nodes", unit="node")
        if HAS_TQDM
        else structure
    )
    for node in top_nodes:
        _render_node_md(node, depth=0, lines=lines)

    return "\n".join(lines)


def _save_result(result: dict, base_name: str, output_dir: str, fmt: str) -> str:
    """Persist the indexing result to disk in the requested format.

    Args:
        result:     The tree structure dict.
        base_name:  Stem of the source file (used for the output filename).
        output_dir: Directory to write the output file (created if absent).
        fmt:        ``"json"`` (default) or ``"markdown"``.

    Returns:
        Absolute path to the written output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if fmt == "markdown":
        output_file = os.path.join(output_dir, f"{base_name}_structure.md")
        content = structure_to_markdown(result)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        output_file = os.path.join(output_dir, f"{base_name}_structure.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return os.path.abspath(output_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process a PDF or Markdown document and generate a PageIndex tree structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python run_pageindex.py --pdf_path report.pdf
  python run_pageindex.py --md_path notes.md --output-format markdown
  python run_pageindex.py --pdf_path paper.pdf --model anthropic/claude-sonnet-4-6 --verbose
  python run_pageindex.py --pdf_path paper.pdf --output-dir ./my_results
""",
    )

    # ---- Input ----
    parser.add_argument("--pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--md_path", type=str, help="Path to the Markdown file")

    # ---- Model ----
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "LiteLLM model string (overrides config.yaml). "
            "Examples: gpt-4o-2024-11-20, anthropic/claude-sonnet-4-6, "
            "gemini/gemini-2.0-flash"
        ),
    )

    # ---- PDF-specific ----
    parser.add_argument(
        "--toc-check-pages", type=int, default=None,
        help="Number of pages to scan for a table of contents (PDF only, default: 20)",
    )
    parser.add_argument(
        "--max-pages-per-node", type=int, default=None,
        help="Maximum pages per tree node (PDF only, default: 10)",
    )
    parser.add_argument(
        "--max-tokens-per-node", type=int, default=None,
        help="Maximum tokens per tree node (PDF only, default: 20000)",
    )

    # ---- Feature flags ----
    parser.add_argument(
        "--if-add-node-id", type=str, default=None,
        help='Add numeric IDs to each node ("yes"/"no")',
    )
    parser.add_argument(
        "--if-add-node-summary", type=str, default=None,
        help='Generate LLM summaries for each node ("yes"/"no")',
    )
    parser.add_argument(
        "--if-add-doc-description", type=str, default=None,
        help='Generate a one-sentence document description ("yes"/"no")',
    )
    parser.add_argument(
        "--if-add-node-text", type=str, default=None,
        help='Embed raw page text in each node ("yes"/"no")',
    )

    # ---- Markdown-specific ----
    parser.add_argument(
        "--if-thinning", type=str, default="no",
        help='Apply tree thinning for sparse Markdown docs ("yes"/"no", Markdown only)',
    )
    parser.add_argument(
        "--thinning-threshold", type=int, default=5000,
        help="Minimum token count threshold for thinning (Markdown only, default: 5000)",
    )
    parser.add_argument(
        "--summary-token-threshold", type=int, default=200,
        help=(
            "Node token count above which a summary is generated "
            "(Markdown only, default: 200)"
        ),
    )

    # ---- Output ----
    parser.add_argument(
        "--output-format",
        type=str, default="json", choices=["json", "markdown"],
        help=(
            'Output format (default: "json"). '
            '"json" writes a structured tree; '
            '"markdown" writes a human-readable outline with headings and summaries.'
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to write the output file (default: ./results)",
    )

    # ---- Logging ----
    parser.add_argument(
        "--verbose", action="store_true",
        help=(
            "Enable verbose logging: shows LiteLLM requests, retries, "
            "and token counts during processing."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    # --- Configure logging ---
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
        import litellm as _litellm
        _litellm.set_verbose = True
    else:
        logging.basicConfig(level=logging.WARNING)

    # --- Validate input ---
    if not args.pdf_path and not args.md_path:
        parser.error("Either --pdf_path or --md_path must be specified.")
    if args.pdf_path and args.md_path:
        parser.error("Only one of --pdf_path or --md_path can be specified.")

    # --- Validate API key before doing any work ---
    _check_api_key(args.model)

    # -----------------------------------------------------------------------
    # PDF processing
    # -----------------------------------------------------------------------
    if args.pdf_path:
        if not args.pdf_path.lower().endswith(".pdf"):
            parser.error("--pdf_path must point to a .pdf file.")
        if not os.path.isfile(args.pdf_path):
            parser.error(f"File not found: {args.pdf_path}")

        user_opt = {
            "model": args.model,
            "toc_check_page_num": args.toc_check_pages,
            "max_page_num_each_node": args.max_pages_per_node,
            "max_token_num_each_node": args.max_tokens_per_node,
            "if_add_node_id": args.if_add_node_id,
            "if_add_node_summary": args.if_add_node_summary,
            "if_add_doc_description": args.if_add_doc_description,
            "if_add_node_text": args.if_add_node_text,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        print(f"Processing PDF: {args.pdf_path}")
        result = page_index_main(args.pdf_path, opt)
        base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]

    # -----------------------------------------------------------------------
    # Markdown processing
    # -----------------------------------------------------------------------
    elif args.md_path:
        if not args.md_path.lower().endswith((".md", ".markdown")):
            parser.error("--md_path must point to a .md or .markdown file.")
        if not os.path.isfile(args.md_path):
            parser.error(f"File not found: {args.md_path}")

        user_opt = {
            "model": args.model,
            "if_add_node_summary": args.if_add_node_summary,
            "if_add_doc_description": args.if_add_doc_description,
            "if_add_node_text": args.if_add_node_text,
            "if_add_node_id": args.if_add_node_id,
        }
        opt = ConfigLoader().load({k: v for k, v in user_opt.items() if v is not None})

        print(f"Processing Markdown: {args.md_path}")
        result = asyncio.run(
            md_to_tree(
                md_path=args.md_path,
                if_thinning=args.if_thinning.lower() == "yes",
                min_token_threshold=args.thinning_threshold,
                if_add_node_summary=opt.if_add_node_summary,
                summary_token_threshold=args.summary_token_threshold,
                model=opt.model,
                if_add_doc_description=opt.if_add_doc_description,
                if_add_node_text=opt.if_add_node_text,
                if_add_node_id=opt.if_add_node_id,
            )
        )
        base_name = os.path.splitext(os.path.basename(args.md_path))[0]

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    print("Parsing complete. Saving output...")
    output_file = _save_result(result, base_name, args.output_dir, args.output_format)
    print(f"Saved to: {output_file}")
