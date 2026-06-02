"""
Incremental Markdown Update with PageIndex - Demo

Shows how PageIndexClient resolves a document by file path: the first run
indexes it fresh; later runs find the same doc_id and call update(), which
re-summarizes only the sections whose content changed.

Flow:
  - First run for a path → index() builds the tree fresh.
  - Later runs → same doc_id is found, update() runs; with no content change
    it reports "unchanged" (zero LLM work).

The source document (documents/sample.md) is copied into the workspace under a
stable path, so re-running the demo reuses the same doc_id. An API key is
required to generate section summaries.

Run:
  python examples/incremental_update_demo.py
"""
import shutil
from pathlib import Path

from pageindex import PageIndexClient

SOURCE_MD = Path(__file__).parent / "documents" / "sample.md"


def ingest_or_update(client, doc_path):
    """Index the doc if new, otherwise incrementally update it."""
    doc_id = client.get_doc_id_by_path(str(doc_path))
    if doc_id:
        result = client.update(doc_id)
        if result.get("status") == "unchanged":
            print(f"\n[{doc_path.name}] Loaded from cache (unchanged): {doc_id}")
        else:
            print(f"\n[{doc_path.name}] Incremental update done: {result}")
    else:
        doc_id = client.index(str(doc_path))
        print(f"\n[{doc_path.name}] Indexed fresh. doc_id: {doc_id}")
    return doc_id


def main():
    workspace = Path(__file__).parent / "workspace"
    client = PageIndexClient(workspace=str(workspace))

    # Stable copy inside the workspace so re-runs reuse the same doc_id.
    workspace.mkdir(parents=True, exist_ok=True)
    md_path = workspace / SOURCE_MD.name
    shutil.copy(SOURCE_MD, md_path)  

    print("== Ingest or update ==")
    doc_id = ingest_or_update(client, md_path)




if __name__ == "__main__":
    main()
