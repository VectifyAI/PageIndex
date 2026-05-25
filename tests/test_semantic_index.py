import sys
from pathlib import Path

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
