import json
from types import SimpleNamespace

import pytest


class SummaryBackend:
    def __init__(self, document_id):
        self.document_id = document_id
        self.calls = []

    def available_channels(self):
        return ("summary",)

    def search_channel(self, channel, query, *, limit=10, filters=None):
        self.calls.append((channel, query, filters))
        return [
            SimpleNamespace(
                document_id=self.document_id,
                snippet=f"summary candidate: {query}",
            )
        ]


class ChannelBackend:
    def __init__(self, document_id, channels=("summary", "entity", "relation")):
        self.document_id = document_id
        self.channels = channels

    def available_channels(self):
        return self.channels

    def search_channel(self, channel, query, *, limit=10, filters=None):
        return [
            SimpleNamespace(
                document_id=self.document_id,
                snippet=f"{channel} candidate: {query}",
            )
        ]


def test_semantic_search_scope_keeps_ordinary_folders_out_of_source_type_filters(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": "Federal Reserve annual report summary"}
            )

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
    )
    filesystem.register_file(
        storage_uri="file:///tmp/report.pdf",
        source_path="examples/documents/report.pdf",
        folder_path="/documents",
        external_id="dsid_report",
        title="report.pdf",
        metadata={"source_type": "examples-documents"},
        content="Federal Reserve supervision and regulation annual report.",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )
    backend = SummaryBackend("dsid_report")
    filesystem.semantic_retrieval_backend = backend
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(
        executor.execute('search-summary "Federal Reserve annual report" /documents')
    )

    assert backend.calls[0][2] == {}
    assert result["data"]["data"][0] == {
        "path": "/documents/report.pdf",
        "summary": "Federal Reserve annual report summary",
        "line_text": "1: Federal Reserve supervision and regulation annual report.",
    }

    executor.json_output = False
    rendered = executor.execute('search-summary "Federal Reserve annual report" /documents')
    assert "path: /documents/report.pdf" in rendered
    assert "summary: Federal Reserve annual report summary" in rendered
    assert "line_text: 1: Federal Reserve supervision and regulation annual report." in rendered
    assert "id=dsid_report" not in rendered
    assert "file_ref=" not in rendered


def test_entity_relation_search_return_minimal_fields_with_summary(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult

    class MetadataGenerator:
        def generate(self, document, *, fields):
            values = {
                "summary": "Risk and compliance summary",
                "entity": "Federal Reserve; Disney",
                "relation": "Federal Reserve affects Disney valuation",
            }
            return MetadataGenerationResult(values={field: values[field] for field in fields})

    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=MetadataGenerator(),
    )
    filesystem.register_file(
        storage_uri="file:///tmp/market-note.pdf",
        source_path="examples/documents/market-note.pdf",
        folder_path="/documents",
        external_id="dsid_market_note",
        title="market-note.pdf",
        content="Federal Reserve policy affects Disney valuation.",
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
                "entity": True,
                "relation": True,
            }
        },
    )
    filesystem.semantic_retrieval_backend = ChannelBackend("dsid_market_note")
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    entity = json.loads(executor.execute('search-entity "Federal Reserve" /documents'))
    assert entity["data"]["data"][0] == {
        "path": "/documents/market-note.pdf",
        "summary": "Risk and compliance summary",
        "line_text": "1: Federal Reserve policy affects Disney valuation.",
        "entity": "Federal Reserve; Disney",
    }

    relation = json.loads(executor.execute('search-relation "Disney valuation" /documents'))
    assert relation["data"]["data"][0] == {
        "path": "/documents/market-note.pdf",
        "summary": "Risk and compliance summary",
        "line_text": "1: Federal Reserve policy affects Disney valuation.",
        "relation": "Federal Reserve affects Disney valuation",
    }

    executor.json_output = False
    rendered = executor.execute('search-entity "Federal Reserve" /documents')
    assert "path: /documents/market-note.pdf" in rendered
    assert "summary: Risk and compliance summary" in rendered
    assert "entity: Federal Reserve; Disney" in rendered
    assert "file_ref=" not in rendered


def test_semantic_search_rejects_unquoted_multi_word_query(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem
    from pageindex.filesystem.commands import PIFSCommandError

    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.register_file(
        storage_uri="file:///tmp/report.pdf",
        source_path="examples/documents/report.pdf",
        folder_path="/documents",
        external_id="dsid_report",
        title="Annual report",
        content="Federal Reserve supervision and regulation annual report.",
    )
    filesystem.semantic_retrieval_backend = SummaryBackend("dsid_report")
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    with pytest.raises(PIFSCommandError, match="Quote multi-word queries"):
        executor.execute("search-summary Federal Reserve /documents")

    with pytest.raises(PIFSCommandError, match="quote it"):
        executor.execute("search-summary Federal Reserve")

    with pytest.raises(PIFSCommandError, match="does not support regex alternation"):
        executor.execute('search-summary "Federal|Reserve" /documents')


def test_semantic_search_scope_filters_explicit_source_type_facets():
    from pageindex.filesystem import PageIndexFileSystem

    assert PageIndexFileSystem._semantic_filters_for_scope(
        {"folder_path": "/source_type=google-drive"}
    ) == {"source_type": "google_drive"}
    assert PageIndexFileSystem._semantic_filters_for_scope(
        {"folder_path": "/semantic/source_type=google-drive"}
    ) == {"source_type": "google_drive"}
    assert PageIndexFileSystem._semantic_filters_for_scope(
        {"folder_path": "/documents"}
    ) == {}


def test_grep_source_file_requires_terms_on_same_line(tmp_path):
    from pageindex.filesystem import PIFSCommandExecutor, PageIndexFileSystem

    source_dir = tmp_path / "source" / "documents"
    source_dir.mkdir(parents=True)
    source = source_dir / "split.json"
    source.write_text(
        '{\n  "first": "alpha evidence lives here",\n'
        '  "second": "omega evidence lives there"\n}\n',
        encoding="utf-8",
    )
    filesystem = PageIndexFileSystem(workspace=tmp_path / "workspace")
    filesystem.register_file(
        storage_uri=str(source),
        source_path="documents/split.json",
        folder_path="/documents",
        external_id="doc_split_terms",
        title="Split source terms",
        content="registered artifact without the searched tokens",
    )
    executor = PIFSCommandExecutor(filesystem, json_output=True)

    result = json.loads(executor.execute('grep -R "alpha omega" /documents'))

    assert result["data"]["mode"] == "files"
    assert result["data"]["data"] == []

    matched = json.loads(executor.execute('grep -R "alpha evidence" /documents'))

    assert matched["data"]["data"][0]["external_id"] == "doc_split_terms"
    assert matched["data"]["data"][0]["line"] == 2
    assert "alpha evidence" in matched["data"]["data"][0]["text"]


def test_existing_summary_projection_index_configures_retrieval_backend(tmp_path, monkeypatch):
    from pageindex.filesystem import PageIndexFileSystem
    from pageindex.filesystem.semantic_index import SemanticIndexRecord, SQLiteVecSemanticIndex

    workspace = tmp_path / "workspace"
    index_dir = workspace / "artifacts" / "projection_indexes"
    summary_index = SQLiteVecSemanticIndex(index_dir / "summary_only_vector.sqlite")
    summary_index.reset(
        dimension=3,
        metadata={
            "channel": "summary",
            "embedding_provider": "openai",
            "embedding_model": "test-embedding",
            "embedding_dimensions": 3,
        },
    )
    summary_index.upsert_many(
        [
            SemanticIndexRecord(
                file_ref="file_a",
                external_id="doc_a",
                source_type="documents",
                source_path="documents/a.pdf",
                title="A",
                text="summary",
                vector=[1.0, 0.0, 0.0],
            )
        ]
    )
    filesystem = PageIndexFileSystem(workspace)
    calls = []

    def fake_configure(index_dir_arg, **kwargs):
        calls.append((index_dir_arg, kwargs))
        filesystem.semantic_retrieval_backend = SummaryBackend("doc_a")
        return filesystem.semantic_retrieval_backend

    monkeypatch.setattr(
        filesystem,
        "configure_hybrid_projection_retrieval",
        fake_configure,
    )

    assert filesystem.configure_existing_projection_retrieval() is True
    assert calls == [
        (
            filesystem.summary_projection_index_dir,
            {
                "embedding_provider": "openai",
                "embedding_model": "test-embedding",
                "embedding_dimensions": 3,
                "embedding_timeout": 60,
            },
        )
    ]
    assert filesystem.semantic_retrieval_channels() == ("summary",)


def test_default_semantic_search_uses_summary_projection_when_only_summary_available(tmp_path):
    from pageindex.filesystem import PageIndexFileSystem
    from pageindex.filesystem.hybrid_projection import HybridProjectionSearchBackend
    from pageindex.filesystem.metadata_generation import MetadataGenerationResult
    from pageindex.filesystem.projection_indexing import SummaryProjectionIndexer

    class FixedEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    class SummaryGenerator:
        def generate(self, document, *, fields):
            return MetadataGenerationResult(
                values={"summary": "vendor renewal risk matrix"}
            )

    source = tmp_path / "source.txt"
    source.write_text("ordinary fixture body", encoding="utf-8")
    index_dir = tmp_path / "workspace" / "artifacts" / "projection_indexes"
    indexer = SummaryProjectionIndexer(
        index_dir,
        embedder=FixedEmbedder(),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=3,
    )
    backend = HybridProjectionSearchBackend(
        index_dir,
        embedder=FixedEmbedder(),
        embedding_provider="test",
        embedding_model="fake",
        embedding_dimensions=3,
    )
    filesystem = PageIndexFileSystem(
        workspace=tmp_path / "workspace",
        metadata_generator=SummaryGenerator(),
        summary_projection_indexer=indexer,
        semantic_retrieval_backend=backend,
    )
    filesystem.register_file(
        storage_uri=source.as_uri(),
        source_path="docs/source.txt",
        folder_path="/documents",
        external_id="doc_summary_only",
        title="Operations note",
        content=source.read_text(encoding="utf-8"),
        metadata={"department": "ops"},
        metadata_policy={
            "fields": {
                "summary": True,
                "doc_type": False,
                "domain": False,
                "topic": False,
            }
        },
    )

    assert filesystem.search("purchase order exposure", semantic=False) == []

    results = filesystem.search("purchase order exposure", semantic=True)

    assert [result.external_id for result in results] == ["doc_summary_only"]
    assert results[0].snippet == "summary_vector rank=1"
