from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._embedding_identity import (
    DEFAULT_OPENAI_BASE_URL,
    normalize_base_url,
    normalize_model,
)
from ._projection_topology import projection_database_pair, projection_database_paths
from ._sqlite_schema import (
    normalized_table_sql,
    regular_table_names,
    sqlite_schema_signature,
)
from .core import DEFAULT_EMBEDDING_DIMENSIONS
from .semantic_index import (
    SCHEMA_VERSION,
    SQLiteVecSemanticIndex,
    SemanticIndexRecord,
)


SUMMARY_INDEX_NAME = "summary"
_EmbeddingCacheKey = tuple[str, str, int, str]


@dataclass(frozen=True)
class SummaryEmbeddingProfile:
    base_url: str | None = DEFAULT_OPENAI_BASE_URL
    model: str = "text-embedding-3-small"
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    timeout: float = 60
    api_key: str | None = None

    def __post_init__(self) -> None:
        model = normalize_model(self.model)
        if int(self.dimensions) <= 0:
            raise ValueError("embedding dimensions must be positive")
        if float(self.timeout) <= 0:
            raise ValueError("embedding timeout must be positive")
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "dimensions", int(self.dimensions))
        object.__setattr__(self, "timeout", float(self.timeout))

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "dimensions": self.dimensions,
        }


@dataclass(frozen=True)
class SummaryProjectionCandidate:
    file_ref: str
    distance: float
    source_type: str
    title: str
    metadata: dict[str, Any]

    @property
    def similarity(self) -> float:
        return 1.0 / (1.0 + max(0.0, self.distance))


class SummaryProjection:
    """Own the complete PIFS Summary Projection lifecycle."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        profile: SummaryEmbeddingProfile,
        embedder: Any | None = None,
        create: bool = False,
        fetch_multiplier: int = 100,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser()
        self.profile = profile
        self._embedder_instance = embedder
        self.fetch_multiplier = fetch_multiplier
        self.index = SQLiteVecSemanticIndex(
            self.index_dir / f"{SUMMARY_INDEX_NAME}.sqlite"
        )
        cache_path = self.index_dir / "embedding_cache.sqlite"
        database_pair = projection_database_pair(self.index_dir)
        if database_pair is not None:
            self.index.validate(self.profile.identity)
            self.embedding_cache = EmbeddingCache(cache_path, create=False)
        elif create:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.index.reset(
                    dimension=self.profile.dimensions,
                    metadata=self.profile.identity,
                )
                self.embedding_cache = EmbeddingCache(cache_path, create=True)
            except Exception:
                self._cleanup_failed_create()
                raise
        else:
            raise RuntimeError("PIFS Summary Projection is not available")

    def _cleanup_failed_create(self) -> None:
        for path in projection_database_paths(self.index_dir):
            for suffix in ("", "-journal", "-shm", "-wal"):
                try:
                    Path(f"{path}{suffix}").unlink()
                except FileNotFoundError:
                    continue

    def upsert_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = str((record.get("metadata") or {}).get("summary") or "").strip()
        if not summary:
            return {"status": "skipped", "reason": "missing_summary"}
        vector = self.embedding_cache.embed_texts(
            [summary],
            profile=self.profile,
            embedder=self._embedder(),
            batch_size=1,
        )[0]
        count = self.index.upsert_many(
            [
                SemanticIndexRecord(
                    file_ref=str(record["file_ref"]),
                    vector=vector,
                    text=summary,
                    external_id=record.get("external_id"),
                    source_type=str(record.get("source_type") or ""),
                    title=str(record.get("title") or ""),
                    metadata=dict(record.get("metadata") or {}),
                )
            ]
        )
        return {
            "status": "ready",
            "indexed_rows": count,
            "index_path": str(self.index.db_path),
            **self.profile.identity,
        }

    def delete_summary(self, file_ref: str) -> int:
        return self.index.delete_file_refs([file_ref])

    def cache_keys_for_records(
        self,
        records: list[dict[str, Any]],
    ) -> set[_EmbeddingCacheKey]:
        summaries = [
            str((record.get("metadata") or {}).get("summary") or "").strip()
            for record in records
        ]
        return self.embedding_cache.keys_for_texts(
            [summary for summary in summaries if summary],
            profile=self.profile,
        )

    def existing_cache_keys(
        self,
        keys: set[_EmbeddingCacheKey],
    ) -> set[_EmbeddingCacheKey]:
        return self.embedding_cache.existing_keys(keys)

    def delete_cache_keys(self, keys: set[_EmbeddingCacheKey]) -> int:
        return self.embedding_cache.delete_keys(keys)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        file_refs: list[str] | None = None,
    ) -> list[SummaryProjectionCandidate]:
        query = normalize_text(query)
        if not query or not self.available:
            return []
        vector = self.embedding_cache.embed_texts(
            [query],
            profile=self.profile,
            embedder=self._embedder(),
            batch_size=1,
        )[0]
        filters = {"file_ref": file_refs} if file_refs is not None else None
        return [
            SummaryProjectionCandidate(
                file_ref=result.file_ref,
                distance=result.distance,
                source_type=result.source_type,
                title=result.title,
                metadata=result.metadata,
            )
            for result in self.index.search(
                vector,
                limit=limit,
                filters=filters,
                fetch_multiplier=self.fetch_multiplier,
            )
        ]

    @property
    def available(self) -> bool:
        return int(self.index.info().get("document_count") or 0) > 0

    def info(self) -> dict[str, Any]:
        return {
            **self.index.info(),
            "embedding_identity": self.profile.identity,
            "available": self.available,
        }

    def _embedder(self) -> Any:
        if self._embedder_instance is None:
            self._embedder_instance = EmbeddingClient(self.profile)
        return self._embedder_instance


class EmbeddingCache:
    def __init__(self, db_path: Path, *, create: bool) -> None:
        self.db_path = db_path
        exists = self.db_path.exists()
        if exists:
            self._validate()
        elif create:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self.connect() as connection:
                self._create_schema(connection)
        else:
            raise RuntimeError(
                "PIFS embedding cache is missing; migrate this workspace with "
                "pifs-data/scripts/migrate_pifs_workspace.py before opening it."
            )

    def connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if read_only:
            connection = sqlite3.connect(
                f"{self.db_path.resolve().as_uri()}?mode=ro",
                uri=True,
            )
        else:
            connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def embed_texts(
        self,
        texts: list[str],
        *,
        profile: SummaryEmbeddingProfile,
        embedder: Any,
        batch_size: int,
    ) -> list[list[float]]:
        hashes = [SQLiteVecSemanticIndex.text_hash(text) for text in texts]
        cached: dict[str, list[float]] = {}
        with self.connect() as connection:
            for text_hash in sorted(set(hashes)):
                row = connection.execute(
                    """
                    SELECT vector_blob
                    FROM embedding_cache
                    WHERE base_url = ? AND model = ? AND dimensions = ? AND text_hash = ?
                    """,
                    (
                        profile.base_url,
                        profile.model,
                        profile.dimensions,
                        text_hash,
                    ),
                ).fetchone()
                if row is not None:
                    cached[text_hash] = decode_vector(
                        bytes(row["vector_blob"]), profile.dimensions
                    )
        missing_positions = [
            index for index, text_hash in enumerate(hashes) if text_hash not in cached
        ]
        for start in range(0, len(missing_positions), max(1, batch_size)):
            positions = missing_positions[start : start + max(1, batch_size)]
            batch_texts = [texts[index] for index in positions]
            vectors = embed_with_retry(embedder, batch_texts)
            if len(vectors) != len(positions):
                raise ValueError(
                    "embedding response length mismatch: "
                    f"requested {len(positions)}, received {len(vectors)}"
                )
            for vector in vectors:
                if len(vector) != profile.dimensions:
                    raise ValueError(
                        "embedding dimension mismatch: "
                        f"expected {profile.dimensions}, received {len(vector)}"
                    )
            with self.connect() as connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_cache(
                        base_url, model, dimensions, text_hash, vector_blob
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            profile.base_url,
                            profile.model,
                            profile.dimensions,
                            hashes[index],
                            encode_vector(vector),
                        )
                        for index, vector in zip(positions, vectors)
                    ],
                )
            for index, vector in zip(positions, vectors):
                cached[hashes[index]] = vector
        return [cached[text_hash] for text_hash in hashes]

    @staticmethod
    def keys_for_texts(
        texts: list[str],
        *,
        profile: SummaryEmbeddingProfile,
    ) -> set[_EmbeddingCacheKey]:
        return {
            (
                str(profile.base_url),
                profile.model,
                profile.dimensions,
                SQLiteVecSemanticIndex.text_hash(text),
            )
            for text in texts
        }

    def existing_keys(
        self,
        keys: set[_EmbeddingCacheKey],
    ) -> set[_EmbeddingCacheKey]:
        existing: set[_EmbeddingCacheKey] = set()
        with self.connect(read_only=True) as connection:
            for key in sorted(keys):
                if connection.execute(
                    """
                    SELECT 1
                    FROM embedding_cache
                    WHERE base_url = ? AND model = ? AND dimensions = ? AND text_hash = ?
                    """,
                    key,
                ).fetchone() is not None:
                    existing.add(key)
        return existing

    def delete_keys(self, keys: set[_EmbeddingCacheKey]) -> int:
        if not keys:
            return 0
        with self.connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """
                DELETE FROM embedding_cache
                WHERE base_url = ? AND model = ? AND dimensions = ? AND text_hash = ?
                """,
                sorted(keys),
            )
            return connection.total_changes - before

    def _validate(self) -> None:
        try:
            with self.connect(read_only=True) as connection:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                actual = self._schema_signature(connection)
                invalid_rows = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM embedding_cache "
                        "WHERE dimensions <= 0 OR length(vector_blob) != dimensions * 4 "
                        "OR trim(base_url) = '' OR trim(model) = '' OR trim(text_hash) = ''"
                    ).fetchone()[0]
                )
                identities = connection.execute(
                    "SELECT DISTINCT base_url, model FROM embedding_cache"
                ).fetchall()
            with sqlite3.connect(":memory:") as expected_connection:
                expected_connection.row_factory = sqlite3.Row
                self._create_schema(expected_connection)
                expected = self._schema_signature(expected_connection)
            invalid_identity = any(
                normalize_base_url(row["base_url"]) != row["base_url"]
                or normalize_model(row["model"]) != row["model"]
                for row in identities
            )
        except (sqlite3.Error, ValueError) as exc:
            raise self._incompatible_schema_error() from exc
        if (
            version != SCHEMA_VERSION
            or actual != expected
            or invalid_rows
            or invalid_identity
        ):
            raise self._incompatible_schema_error()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            f"""
            CREATE TABLE embedding_cache (
                base_url TEXT NOT NULL,
                model TEXT NOT NULL,
                dimensions INTEGER NOT NULL CHECK(dimensions > 0),
                text_hash TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(base_url, model, dimensions, text_hash)
            );
            PRAGMA user_version = {SCHEMA_VERSION};
            """
        )

    @staticmethod
    def _schema_signature(connection: sqlite3.Connection) -> dict[str, Any]:
        tables = regular_table_names(connection)
        if tables != {"embedding_cache"}:
            return {"tables": tables}
        signature = sqlite_schema_signature(connection, tables)
        signature["sql"] = normalized_table_sql(connection, "embedding_cache")
        return signature

    @staticmethod
    def _incompatible_schema_error() -> RuntimeError:
        return RuntimeError(
            "Incompatible PIFS embedding cache schema; migrate this workspace with "
            "pifs-data/scripts/migrate_pifs_workspace.py before opening it."
        )


def validate_projection_topology(index_dir: str | Path) -> bool:
    index_dir = Path(index_dir).expanduser()
    database_pair = projection_database_pair(index_dir)
    if database_pair is None:
        return False
    summary_path, cache_path = database_pair
    SQLiteVecSemanticIndex(summary_path).validate_schema()
    EmbeddingCache(cache_path, create=False)
    return True


class EmbeddingClient:
    def __init__(self, profile: SummaryEmbeddingProfile) -> None:
        from openai import OpenAI

        if not profile.api_key:
            raise ValueError("embedding_api_key is required for PIFS embeddings")
        self.profile = profile
        self.client = OpenAI(
            api_key=profile.api_key,
            base_url=profile.base_url,
            timeout=profile.timeout,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.profile.model,
            input=texts,
            dimensions=self.profile.dimensions,
        )
        return [
            list(item.embedding)
            for item in sorted(response.data, key=lambda item: item.index)
        ]


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def embed_with_retry(
    embedder: Any,
    texts: list[str],
    *,
    max_attempts: int = 8,
) -> list[list[float]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed(texts)
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_embedding_error(exc):
                raise
            time.sleep(min(120.0, 2.0 ** (attempt - 1)))
    raise RuntimeError("unreachable embedding retry state")


def is_retryable_embedding_error(exc: Exception) -> bool:
    retryable = getattr(exc, "retryable", None)
    if isinstance(retryable, bool):
        return retryable
    status_code = getattr(exc, "status_code", None)
    try:
        status = int(status_code)
    except (TypeError, ValueError):
        status = None
    if status is not None:
        if status in {408, 409, 429} or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    name = exc.__class__.__name__.lower()
    return any(token in name for token in ("timeout", "connection", "ratelimit"))


def encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes, dimensions: int) -> list[float]:
    if len(blob) != dimensions * 4:
        raise ValueError(
            f"cached embedding has {len(blob) // 4} dimensions, expected {dimensions}"
        )
    return list(struct.unpack(f"<{dimensions}f", blob))
