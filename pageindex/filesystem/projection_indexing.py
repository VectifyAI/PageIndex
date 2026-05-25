from __future__ import annotations

from pathlib import Path
from typing import Any

from .hybrid_projection import (
    EmbeddingCache,
    INDEX_BY_CHANNEL,
    embedding_cache_model_key,
    make_embedder,
)
from .semantic_index import SQLiteVecSemanticIndex, SemanticIndexRecord


class SummaryProjectionIndexer:
    """Synchronous register-time summary projection indexer."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedder: Any,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int = 256,
        embedding_cache_path: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser()
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.cache_model = embedding_cache_model_key(embedding_model, embedding_dimensions)
        self.embedding_cache = EmbeddingCache(
            Path(embedding_cache_path).expanduser()
            if embedding_cache_path is not None
            else self.index_dir / "embedding_cache.sqlite"
        )
        self.index = SQLiteVecSemanticIndex(
            self.index_dir / f"{INDEX_BY_CHANNEL['summary']}.sqlite"
        )
        self._ensure_index()

    @classmethod
    def from_provider(
        cls,
        index_dir: str | Path,
        *,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimensions: int = 256,
        embedding_timeout: float = 60,
        **kwargs: Any,
    ) -> "SummaryProjectionIndexer":
        return cls(
            index_dir,
            embedder=make_embedder(
                embedding_provider,
                embedding_model,
                dimensions=embedding_dimensions,
                timeout=embedding_timeout,
            ),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            **kwargs,
        )

    def upsert_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = str((record.get("derived_metadata") or {}).get("summary") or "").strip()
        if not summary:
            return {"status": "skipped", "reason": "missing_summary"}
        vector = self.embedding_cache.embed_texts(
            [summary],
            provider=self.embedding_provider,
            model=self.cache_model,
            embedder=self.embedder,
            batch_size=1,
        )[0]
        metadata = dict(record.get("metadata") or {})
        metadata.update(record.get("derived_metadata") or {})
        count = self.index.upsert_many(
            [
                SemanticIndexRecord(
                    file_ref=str(record["file_ref"]),
                    vector=vector,
                    text=summary,
                    external_id=record.get("external_id"),
                    source_type=str(record.get("source_type") or ""),
                    source_path=str(record.get("source_path") or ""),
                    title=str(record.get("title") or ""),
                    metadata=metadata,
                )
            ]
        )
        return {
            "status": "ready",
            "indexed_rows": count,
            "index_path": str(self.index.db_path),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }

    def _ensure_index(self) -> None:
        if not self.index.db_path.exists():
            self.index.reset(
                dimension=self.embedding_dimensions,
                metadata=self._index_metadata(),
            )
            return
        try:
            if self.index.dimension() != self.embedding_dimensions:
                self.index.reset(
                    dimension=self.embedding_dimensions,
                    metadata=self._index_metadata(),
                )
        except Exception:
            self.index.reset(
                dimension=self.embedding_dimensions,
                metadata=self._index_metadata(),
            )

    def _index_metadata(self) -> dict[str, Any]:
        return {
            "channel": "summary",
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
        }
