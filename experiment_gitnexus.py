"""PageIndex Experiment 1: Index GitNexus + Search Verification"""
import asyncio
import json
import time
import sys
import os

sys.path.insert(0, "D:/projects/pageindex-local")

from pageindex.code_indexer import CodeIndexer
from pageindex.code_searcher import CodeSearcher

RESULTS_DIR = "D:/projects/pageindex-local/results"
GITNEXUS_SRC = "D:/projects/GitNexus/gitnexus/src"


async def main():
    print("=" * 60)
    print("EXPERIMENT 1: PageIndex Code Indexing + Search on GitNexus")
    print("=" * 60)

    # --- Step 1: Index ---
    print("\n## Step 1: Indexing GitNexus src/")
    t0 = time.time()
    indexer = CodeIndexer(max_files=200)
    repo_index = await indexer.index_directory(
        dir_path=GITNEXUS_SRC,
        language_filter="typescript",
        repo_name="gitnexus-code",
    )
    t1 = time.time()

    total_symbols = sum(len(f.symbols) for f in repo_index.files)
    total_call_edges = sum(len(v) for v in repo_index.call_graph.values())

    print(f"  Files indexed:    {len(repo_index.files)}")
    print(f"  Symbols extracted:{total_symbols}")
    print(f"  Call graph nodes: {len(repo_index.call_graph)}")
    print(f"  Call graph edges: {total_call_edges}")
    print(f"  Languages:        {repo_index.language_composition}")
    print(f"  Time:             {t1 - t0:.2f}s")

    # Save index
    out_path = os.path.join(RESULTS_DIR, "gitnexus-code_code_index.json")
    indexer.save_index(repo_index, out_path)
    fsize = os.path.getsize(out_path)
    print(f"  Saved to:         {out_path} ({fsize / 1024:.1f} KB)")

    # --- Step 2: Search ---
    print("\n## Step 2: Search Queries")
    searcher = CodeSearcher(results_dir=RESULTS_DIR)

    # Query 1: definition — use actual indexed symbol
    print("\n### Query 1: Definition — 'detectFrameworkFromPath'")
    t0 = time.time()
    r1 = await searcher.search(
        query="detectFrameworkFromPath",
        repo_names=["gitnexus-code"],
        mode="definition",
    )
    print(f"  Time: {time.time() - t0:.2f}s")
    print(r1)

    # Query 2: impact — use actual indexed symbol
    print("\n### Query 2: Impact — 'detectFrameworkFromPath'")
    t0 = time.time()
    r2 = await searcher.search(
        query="detectFrameworkFromPath",
        repo_names=["gitnexus-code"],
        mode="impact",
    )
    print(f"  Time: {time.time() - t0:.2f}s")
    print(r2)

    # Query 3: relevance (LLM ranking)
    print("\n### Query 3: Relevance — 'how is symbol change detection implemented'")
    t0 = time.time()
    r3 = await searcher.search(
        query="how is symbol change detection implemented",
        repo_names=["gitnexus-code"],
        mode="relevance",
        top_k=3,
    )
    print(f"  Time: {time.time() - t0:.2f}s")
    print(r3)

    # --- Step 3: Accuracy Assessment ---
    print("\n## Step 3: Accuracy Assessment")
    print("Q1 (definition): Found detectFrameworkFromPath?", "detectFrameworkFromPath" in r1)
    print("Q2 (impact): Has callers?", "Called by" in r2 or "callers" in r2.lower())
    print("Q3 (relevance): Has results?", "###" in r3)

    # Dump symbol list for reference
    print("\n## All Symbol Names (for reference):")
    for fi in repo_index.files:
        for s in fi.symbols:
            print(f"  {fi.file_path}:{s.name} ({s.type}) L{s.line_range[0]}-{s.line_range[1]}")


if __name__ == "__main__":
    asyncio.run(main())
