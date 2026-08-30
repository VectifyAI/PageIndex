"""`books` — manage the personal book library.

    books add BOOK.pdf [--profile diary] [--model claude-cli/sonnet] [--no-summaries] [--force]
    books list [--json]
    books show BOOK [--depth N] [--tier summary|digest]
    books summarize BOOK [--tier summary|digest] [--model M] [--node ID ...] [--force]
    books digest BOOK [--node ID] [--out DIR]          (Task 9)
    books mcp                                          (Task 10)
"""
from __future__ import annotations

import argparse
import json
import sys

from ..local_store import DocStore
from ..utils import structure_to_list
from .config import LibraryConfig
from .ingest import add_book, find_book
from .summaries import TIERS, describe_book, summarize_book


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="books", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", help="library home (default: $BOOKS_HOME or ~/github/repos/book-library)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="index a PDF and generate routing summaries")
    a.add_argument("pdf")
    a.add_argument("--profile", choices=["nonfiction", "diary"])
    a.add_argument("--model")
    a.add_argument("--no-summaries", action="store_true")
    a.add_argument("--force", action="store_true")

    ls = sub.add_parser("list", help="list indexed books")
    ls.add_argument("--json", action="store_true")

    sh = sub.add_parser("show", help="print a book's tree")
    sh.add_argument("book")
    sh.add_argument("--depth", type=int, default=None)
    sh.add_argument("--tier", choices=list(TIERS), default="summary")

    sm = sub.add_parser("summarize", help="generate or resume a summary tier")
    sm.add_argument("book")
    sm.add_argument("--tier", choices=list(TIERS), default="summary")
    sm.add_argument("--model")
    sm.add_argument("--node", action="append", dest="nodes")
    sm.add_argument("--force", action="store_true")

    dg = sub.add_parser("digest", help="write Markdown digest(s) for a book or one node")
    dg.add_argument("book")
    dg.add_argument("--node")
    return p


def _rows(store: DocStore) -> list[dict]:
    rows = []
    for meta in sorted(store.list_metas(), key=lambda m: m.get("createdAt") or ""):
        extra = meta.get("metadata") or {}
        rows.append({"doc_id": meta["id"], "name": meta["name"],
                     "title": extra.get("title"), "profile": extra.get("profile"),
                     "pages": meta.get("pageNum"), "status": meta.get("status"),
                     "nodes": len(structure_to_list(store.get_tree(meta["id"]) or [])),
                     "summary_done": extra.get("summary_tier_done"),
                     "digest_done": extra.get("digest_tier_done"),
                     "description": meta.get("description")})
    return rows


def _print_tree(nodes: list[dict], tier: str, depth: int | None, level: int = 0) -> None:
    for node in nodes:
        text = (node.get(tier) or "").replace("\n", " ")
        snippet = f"  — {text[:90]}…" if len(text) > 90 else (f"  — {text}" if text else "")
        print("  " * level + f"[{node.get('node_id')}] {node.get('title', '')} "
              f"p{node.get('start_index')}-{node.get('end_index')}{snippet}")
        if node.get("nodes") and (depth is None or level + 1 < depth):
            _print_tree(node["nodes"], tier, depth, level + 1)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = LibraryConfig.load(args.home)
    cfg.home.mkdir(parents=True, exist_ok=True)
    store = DocStore(str(cfg.storage_path))
    try:
        if args.command == "add":
            out = add_book(args.pdf, cfg, profile=args.profile, model=args.model,
                           summaries=not args.no_summaries, force=args.force)
            print(f"{out['doc_id']}  {out['name']}  {out['nodes']} nodes  "
                  f"{out['pages']} pages  {out['status']}")
            return 0
        if args.command == "list":
            rows = _rows(store)
            if args.json:
                print(json.dumps(rows, indent=1, ensure_ascii=False))
            else:
                for r in rows:
                    print(f"{r['doc_id']}  {r['name']}  [{r['profile']}]  {r['pages']} pages  "
                          f"{r['nodes']} nodes  {r['status']}  "
                          f"summary={'✓' if r['summary_done'] else '–'} "
                          f"digest={'✓' if r['digest_done'] else '–'}")
            return 0
        meta = find_book(store, args.book)
        if args.command == "show":
            extra = meta.get("metadata") or {}
            print(f"{extra.get('title') or meta['name']}  ({meta['id']}, {meta['pageNum']} pages)")
            if meta.get("description"):
                print(meta["description"])
            _print_tree(store.get_tree(meta["id"]) or [], args.tier, args.depth)
            return 0
        if args.command == "summarize":
            model = args.model or (cfg.digest_model if args.tier == "digest" else cfg.index_model)
            stats = summarize_book(store, meta["id"], tier=args.tier, model=model,
                                   force=args.force, node_ids=args.nodes)
            print(f"{args.tier}: {stats['generated']} generated, {stats['skipped']} skipped, "
                  f"{stats['failed']} failed")
            for err in stats["errors"]:
                print("  " + err, file=sys.stderr)
            if args.tier == "summary" and stats["failed"] == 0 and args.nodes is None:
                describe_book(store, meta["id"], model=model)
                store.update_meta(meta["id"], status="completed")
            return 0 if stats["failed"] == 0 else 2
        if args.command == "digest":
            from .digest import write_digest
            path = write_digest(cfg, store, meta["id"], node_id=args.node)
            print(path)
            return 0
    except (LookupError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
