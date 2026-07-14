import hashlib

import pytest

from pageindex.filesystem.semantic_index import (
    SemanticIndexRecord,
    SQLiteVecSemanticIndex,
)
from pageindex.filesystem.semantic_projection import (
    EmbeddingCache,
    SummaryEmbeddingProfile,
    SummaryProjection,
)


class FixedEmbedder:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def embed(self, texts):
        return [[1.0, *([0.0] * (self.dimensions - 1))] for _ in texts]


def test_sqlite_vec_semantic_index_round_trip(tmp_path):
    index = SQLiteVecSemanticIndex(tmp_path / "semantic.sqlite")
    index.reset(dimension=3, metadata={"kind": "summary"})
    index.upsert_many(
        [
            SemanticIndexRecord(
                file_ref="file_a",
                external_id="doc_a",
                source_type="documents",
                title="Multipart upload limits",
                text="multipart upload limits",
                vector=[1.0, 0.0, 0.0],
                metadata={"topic": "uploads"},
            ),
            SemanticIndexRecord(
                file_ref="file_b",
                external_id="doc_b",
                source_type="documents",
                title="GPU cache issue",
                text="gpu cache issue",
                vector=[0.0, 1.0, 0.0],
                metadata={"topic": "runtime"},
            ),
        ]
    )

    assert [
        item.external_id for item in index.search([0.9, 0.1, 0.0], limit=2)
    ] == ["doc_a", "doc_b"]


def test_sqlite_vec_file_ref_filter_is_not_limited_by_global_rank(tmp_path):
    index = SQLiteVecSemanticIndex(tmp_path / "semantic.sqlite")
    index.reset(dimension=2, metadata={"kind": "summary"})
    records = [
        SemanticIndexRecord(
            file_ref=f"file_off_{item:02d}",
            external_id=f"doc_off_{item:02d}",
            source_type="documents",
            title=f"Off scope {item:02d}",
            text="off scope",
            vector=[1.0, 0.0],
        )
        for item in range(30)
    ]
    records.append(
        SemanticIndexRecord(
            file_ref="file_in_scope",
            external_id="doc_in_scope",
            source_type="documents",
            title="In scope",
            text="in scope",
            vector=[0.0, 1.0],
        )
    )
    index.upsert_many(records)

    results = index.search(
        [1.0, 0.0],
        limit=1,
        filters={"file_ref": ["file_in_scope"]},
    )

    assert [item.file_ref for item in results] == ["file_in_scope"]


def test_summary_projection_owns_write_search_and_delete_lifecycle(tmp_path):
    profile = SummaryEmbeddingProfile(
        base_url="https://EXAMPLE.invalid/v1/",
        model="fake",
        dimensions=3,
        api_key="runtime-only",
    )
    projection = SummaryProjection(
        tmp_path / "projection",
        profile=profile,
        embedder=FixedEmbedder(3),
        create=True,
    )
    record = {
        "file_ref": "file_a",
        "external_id": "doc_a",
        "source_type": "documents",
        "title": "A",
        "metadata": {"summary": "Unified summary", "department": "ops"},
    }

    assert projection.upsert_summary(record)["status"] == "ready"
    assert projection.info()["embedding_identity"] == {
        "base_url": "https://example.invalid/v1",
        "model": "fake",
        "dimensions": 3,
    }
    assert [candidate.file_ref for candidate in projection.search("summary")] == [
        "file_a"
    ]
    assert projection.delete_summary("file_a") == 1
    assert projection.available is False


def test_projection_profile_mismatch_preserves_existing_database(tmp_path):
    index_dir = tmp_path / "projection"
    profile = SummaryEmbeddingProfile(
        base_url="https://example.invalid/v1",
        model="fake",
        dimensions=3,
        api_key="runtime-only",
    )
    projection = SummaryProjection(
        index_dir,
        profile=profile,
        embedder=FixedEmbedder(3),
        create=True,
    )
    projection.upsert_summary(
        {
            "file_ref": "file_a",
            "external_id": "doc_a",
            "source_type": "documents",
            "title": "A",
            "metadata": {"summary": "Preserve me"},
        }
    )
    summary_path = index_dir / "summary.sqlite"
    before = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    mismatch = SummaryEmbeddingProfile(
        base_url="https://example.invalid/v1",
        model="different",
        dimensions=3,
        api_key="runtime-only",
    )

    with pytest.raises(Exception, match="Incompatible PIFS Summary Embedding Profile"):
        SummaryProjection(index_dir, profile=mismatch, create=False)

    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == before


def test_embedding_cache_rejects_response_length_mismatch(tmp_path):
    class ShortEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0, 0.0]]

    cache = EmbeddingCache(tmp_path / "cache.sqlite", create=True)
    profile = SummaryEmbeddingProfile(
        base_url="https://example.invalid/v1",
        model="fake",
        dimensions=3,
        api_key="runtime-only",
    )

    with pytest.raises(ValueError, match="embedding response length mismatch"):
        cache.embed_texts(
            ["first", "second"],
            profile=profile,
            embedder=ShortEmbedder(),
            batch_size=2,
        )


def test_embed_with_retry_only_retries_transient_errors(monkeypatch):
    from pageindex.filesystem.semantic_projection import embed_with_retry

    sleeps = []

    class EmbeddingError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    class FlakyEmbedder:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            if self.calls == 1:
                raise EmbeddingError(500)
            return [[1.0, 0.0, 0.0]]

    monkeypatch.setattr(
        "pageindex.filesystem.semantic_projection.time.sleep", sleeps.append
    )
    embedder = FlakyEmbedder()
    assert embed_with_retry(embedder, ["text"]) == [[1.0, 0.0, 0.0]]
    assert embedder.calls == 2
    assert sleeps == [1.0]

    class PermanentEmbedder:
        calls = 0

        def embed(self, texts):
            self.calls += 1
            raise EmbeddingError(401)

    permanent = PermanentEmbedder()
    with pytest.raises(EmbeddingError):
        embed_with_retry(permanent, ["text"])
    assert permanent.calls == 1
