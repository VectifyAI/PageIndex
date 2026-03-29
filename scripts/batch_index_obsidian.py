"""Batch index Obsidian notes into PageIndex.

Usage:
    python scripts/batch_index_obsidian.py          # index P0 only
    python scripts/batch_index_obsidian.py --all     # index P0 + P1
"""

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from pageindex.page_index_md import md_to_tree
from pageindex.utils import structure_to_list

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

OBS = Path("E:/knowledge/Knowledge")

# P0: must-have for expert agent context
P0 = [
    ("dev-system",          "00-Index/开发系统.md"),
    ("project-panorama",    "01-Projects/项目全景.md"),
    ("ai-trading-loop",     "03-Trading/AI自循环系统.md"),
    ("expert-agent-rag",    "01-Projects/专家Agent-RAG.md"),
    ("dev-architecture",    "00-Index/开发架构.md"),
]

# P1: useful reference
P1 = [
    ("a2a-dev",             "01-Projects/a2a-dev.md"),
    ("super-service-map",   "02-Infrastructure/SUPER-service-map.md"),
    ("agent-memory-survey", "04-Research/Agent记忆系统全景.md"),
    ("cli-anything",        "04-Research/cli-anything-playbook.md"),
    ("review-protocol",     "05-Protocols/复盘流程.md"),
    ("claw-mesh",           "01-Projects/claw-mesh.md"),
    ("nadirclaw-report",    "02-Infrastructure/NadirClaw-Integration-Report.md"),
    ("trading-failures",    "03-Trading/Failures.md"),
    ("vibe-coding",         "04-Research/vibe-coding-critique.md"),
]


async def index_one(doc_name: str, rel_path: str) -> str:
    full = OBS / rel_path
    if not full.exists():
        return f"SKIP {doc_name}: file not found"

    out = RESULTS / f"{doc_name}_structure.json"
    if out.exists():
        return f"SKIP {doc_name}: already indexed"

    t0 = time.time()
    tree = await md_to_tree(
        md_path=str(full),
        if_add_node_summary="yes",
        if_add_node_id="yes",
        model="MiniMax-M2.7-highspeed",
    )
    tree["doc_name"] = doc_name
    tree["source_path"] = str(full)
    out.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    nodes = len(structure_to_list(tree.get("structure", [])))
    elapsed = time.time() - t0
    return f"OK {doc_name}: {nodes} nodes in {elapsed:.0f}s"


async def main():
    do_all = "--all" in sys.argv
    batch = P0 + (P1 if do_all else [])
    print(f"Indexing {len(batch)} docs ({'P0+P1' if do_all else 'P0 only'})...\n")

    for doc_name, rel_path in batch:
        result = await index_one(doc_name, rel_path)
        print(result)

    total = len(list(RESULTS.glob("*_structure.json")))
    print(f"\nTotal indexed: {total} documents")


if __name__ == "__main__":
    asyncio.run(main())
