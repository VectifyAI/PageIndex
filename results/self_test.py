"""PageIndex self-indexing test script."""
import asyncio
import json
import sys
sys.path.insert(0, "D:/projects/pageindex-local")

from pageindex.code_indexer import CodeIndexer
from pageindex.code_searcher import CodeSearcher

TARGET_DIR = "D:/projects/pageindex-local/pageindex"
RESULTS_DIR = "D:/projects/pageindex-local/results"

async def main():
    # Step 1: Index
    indexer = CodeIndexer()
    idx = await indexer.index_directory(TARGET_DIR, repo_name="pageindex")
    indexer.save_index(idx, f"{RESULTS_DIR}/pageindex_code_index.json")

    print("=== INDEXING STATS ===")
    print(f"Files: {idx.total_files}")
    print(f"Symbols: {idx.total_symbols}")
    print(f"Call graph edges: {sum(len(v) for v in idx.call_graph.values())}")
    print(f"Languages: {idx.language_composition}")
    print()

    for fi in idx.files:
        print(f"  {fi.file_path}: {len(fi.symbols)} symbols, {fi.metrics.total_lines} lines")
    print()

    # Step 2: Search tests
    searcher = CodeSearcher(RESULTS_DIR)

    # Definition searches
    for q in ["CodeIndexer", "index_directory", "search"]:
        result = await searcher.search(q, repo_names=["pageindex"], mode="definition")
        print(f"=== DEFINITION: {q} ===")
        print(result)
        print()

    # Impact search
    result = await searcher.search("index_directory", repo_names=["pageindex"], mode="impact")
    print("=== IMPACT: index_directory ===")
    print(result)
    print()

asyncio.run(main())
