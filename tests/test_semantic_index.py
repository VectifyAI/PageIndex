import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pageindex.filesystem.semantic_index import (
    SemanticIndexRecord,
    SQLiteVecSemanticIndex,
)


def test_sqlite_vec_semantic_index_round_trip(tmp_path):
    index = SQLiteVecSemanticIndex(tmp_path / "semantic.sqlite")
    index.reset(dimension=3, metadata={"field_mode": "summary"})

    index.upsert_many(
        [
            SemanticIndexRecord(
                file_ref="file_a",
                external_id="doc_a",
                source_type="github",
                source_path="github/a.json",
                title="Multipart upload limits",
                text="multipart upload limits",
                vector=[1.0, 0.0, 0.0],
                metadata={"topic": "uploads"},
            ),
            SemanticIndexRecord(
                file_ref="file_b",
                external_id="doc_b",
                source_type="slack",
                source_path="slack/b.json",
                title="GPU cache issue",
                text="gpu cache issue",
                vector=[0.0, 1.0, 0.0],
                metadata={"topic": "runtime"},
            ),
        ]
    )

    assert index.info()["document_count"] == 2

    results = index.search([0.9, 0.1, 0.0], limit=2)
    assert [item.external_id for item in results] == ["doc_a", "doc_b"]

    filtered = index.search(
        [0.9, 0.1, 0.0],
        limit=2,
        filters={"source_type": "slack"},
    )
    assert [item.external_id for item in filtered] == ["doc_b"]


def test_summary_projection_indexes_unified_metadata_summary(tmp_path):
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    class FakeEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    indexer = SummaryProjectionIndexer(
        tmp_path / "projection",
        embedder=FakeEmbedder(),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=3,
    )

    result = indexer.upsert_summary(
        {
            "file_ref": "file_a",
            "external_id": "doc_a",
            "source_type": "documents",
            "source_path": "docs/a.pdf",
            "title": "A",
            "metadata": {
                "summary": "Unified metadata summary.",
                "department": "ops",
            },
        }
    )

    assert result["status"] == "ready"
    hits = indexer.index.search([1.0, 0.0, 0.0], limit=1)
    assert hits[0].external_id == "doc_a"
    assert hits[0].metadata["summary"] == "Unified metadata summary."
    assert hits[0].metadata["department"] == "ops"


def test_hash_embedding_provider_is_not_available():
    from pageindex.filesystem.hybrid_projection import make_embedder

    with pytest.raises(ValueError, match="unknown embedding provider: hash"):
        make_embedder("hash", "unused", dimensions=256, timeout=1)
