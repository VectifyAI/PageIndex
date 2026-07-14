from __future__ import annotations

from pathlib import Path
from typing import Any


class FakePageIndexClient:
    def __init__(self):
        self.documents: dict[str, dict[str, Any]] = {}

    def index(self, file_path, mode="auto"):
        path = Path(file_path)
        doc_id = f"pi_{path.stem}_{len(self.documents)}"
        text = path.read_text(encoding="utf-8")
        self.documents[doc_id] = {
            "id": doc_id,
            "type": "md",
            "path": str(path.resolve()),
            "doc_name": path.name,
            "doc_description": f"Summary for {path.name}",
            "line_count": len(text.splitlines()),
            "structure": [{"title": path.stem, "node_id": "0001", "nodes": []}],
            "pages": [{"page": 1, "content": text}],
        }
        return doc_id

    def _ensure_doc_loaded(self, doc_id):
        return None

    def get_document_structure(self, doc_id):
        import json

        return json.dumps(self.documents[doc_id]["structure"])

    def get_page_content(self, doc_id, pages):
        import json

        return json.dumps(self.documents[doc_id]["pages"])


class FakeEmbeddingClient:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * (self.dimensions - 1))] for _ in texts]


def register_markdown(
    filesystem,
    tmp_path: Path,
    external_id: str,
    folder_path: str,
    *,
    title: str | None = None,
    text: str | None = None,
    metadata: dict[str, Any] | None = None,
):
    client = getattr(filesystem, "_test_pageindex_client", None)
    if client is None:
        client = FakePageIndexClient()
        filesystem._test_pageindex_client = client
        filesystem._pageindex_client = lambda: client
    if filesystem.summary_projection is None:
        from pageindex.filesystem.semantic_projection import SummaryProjection

        profile = filesystem._summary_embedding_profile()
        filesystem.summary_projection = SummaryProjection(
            filesystem.summary_projection_index_dir,
            profile=profile,
            embedder=FakeEmbeddingClient(profile.dimensions),
            create=True,
        )
    filename = title if title and title.endswith(".md") else f"{external_id}.md"
    source = tmp_path / filename
    source.write_text(text or f"{external_id} alpha evidence", encoding="utf-8")
    return filesystem.register_file(
        storage_uri=source.as_uri(),
        folder_path=folder_path,
        external_id=external_id,
        title=title or filename,
        content_type="text/markdown",
        content=source.read_text(encoding="utf-8"),
        metadata=metadata or {},
    )
