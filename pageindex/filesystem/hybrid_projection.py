from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .semantic_index import SQLiteVecSemanticIndex, SemanticIndexError, SemanticSearchResult


INDEX_BY_CHANNEL = {
    "metadata": "metadata_composite_vector",
    "summary": "summary_only_vector",
    "entity": "entity_vectors",
    "constraint": "constraint_vectors",
    "relation": "relation_vectors",
}
HYBRID_ENTITY_RELATION_CHANNELS = ("metadata", "entity", "constraint", "relation")
SEMANTIC_TOOL_CHANNELS = ("summary", "entity", "relation")
HYBRID_ENTITY_RELATION_WEIGHTS = {
    "metadata": 0.25,
    "entity": 0.25,
    "relation": 0.30,
    "constraint": 0.20,
}


@dataclass(frozen=True)
class QueryProjection:
    entities: list[str]
    relations: list[str]
    constraints: list[str]
    expected_answer_type: str = ""


@dataclass(frozen=True)
class HybridProjectionCandidate:
    document_id: str
    score: float
    sources: list[dict[str, Any]]
    source_type: str
    source_path: str
    title: str
    metadata: dict[str, Any]
    snippet: str


class HybridProjectionSearchBackend:
    """Hybrid entity/relation/vector retrieval over rebuildable projection indexes.

    The SQLite catalog remains the source of truth. This backend only reads
    external sqlite-vec projection indexes and returns candidate document ids
    for the catalog to resolve and filter.
    """

    def __init__(
        self,
        index_dir: str | Path,
        *,
        embedder: Any,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int = 256,
        embedding_cache_path: str | Path | None = None,
        per_channel_limit: int = 100,
        fetch_multiplier: int = 100,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser()
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
        self.per_channel_limit = per_channel_limit
        self.fetch_multiplier = fetch_multiplier
        self.indexes = {
            channel: SQLiteVecSemanticIndex(self.index_dir / f"{index_name}.sqlite")
            for channel, index_name in INDEX_BY_CHANNEL.items()
        }

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
    ) -> "HybridProjectionSearchBackend":
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

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridProjectionCandidate]:
        query = normalize_text(query)
        if not query:
            return []
        projection = heuristic_query_projection(query)
        channels = tuple(
            channel
            for channel in HYBRID_ENTITY_RELATION_CHANNELS
            if self._channel_document_count(channel) > 0
        )
        if not channels:
            return []
        channel_hits = self._search_channels(
            query=query,
            projection=projection,
            limit=max(limit, self.per_channel_limit),
            filters=filters,
            channels=channels,
        )
        return aggregate_hybrid_entity_relation(channel_hits, projection)[:limit]

    def search_channel(
        self,
        channel: str,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridProjectionCandidate]:
        if channel not in SEMANTIC_TOOL_CHANNELS:
            raise ValueError(f"unsupported semantic channel: {channel}")
        if channel not in self.available_channels():
            return []
        query = normalize_text(query)
        if not query:
            return []
        projection = heuristic_query_projection(query)
        vector = self.embedding_cache.embed_texts(
            [query_text_for_channel(channel, query, projection)],
            provider=self.embedding_provider,
            model=self.cache_model,
            embedder=self.embedder,
            batch_size=1,
        )[0]
        results = self.indexes[channel].search(
            vector,
            limit=limit,
            filters=filters,
            fetch_multiplier=self.fetch_multiplier,
        )
        return rank_single_semantic_channel(channel, results)

    def available_channels(self) -> tuple[str, ...]:
        return tuple(
            channel
            for channel in SEMANTIC_TOOL_CHANNELS
            if self._channel_document_count(channel) > 0
        )

    def info(self) -> dict[str, Any]:
        return {
            "index_dir": str(self.index_dir),
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": self.embedding_dimensions,
            "strategy": "hybrid_entity_relation_vector",
            "available_channels": list(self.available_channels()),
            "channels": {
                channel: self._safe_channel_info(channel)
                for channel in self.indexes
            },
        }

    def _channel_document_count(self, channel: str) -> int:
        info = self._safe_channel_info(channel)
        if not info.get("available"):
            return 0
        return int(info.get("document_count") or 0)

    def _safe_channel_info(self, channel: str) -> dict[str, Any]:
        index = self.indexes[channel]
        if not index.db_path.exists():
            return {
                "db_path": str(index.db_path),
                "available": False,
                "document_count": 0,
                "error": "index file is missing",
            }
        try:
            info = index.info()
        except (OSError, sqlite3.Error, SemanticIndexError) as exc:
            return {
                "db_path": str(index.db_path),
                "available": False,
                "document_count": 0,
                "error": str(exc),
            }
        return {**info, "available": int(info.get("document_count") or 0) > 0}

    def _search_channels(
        self,
        *,
        query: str,
        projection: QueryProjection,
        limit: int,
        filters: dict[str, Any] | None,
        channels: tuple[str, ...],
    ) -> dict[str, list[SemanticSearchResult]]:
        query_texts = {
            channel: query_text_for_channel(channel, query, projection)
            for channel in channels
        }
        vectors = self.embedding_cache.embed_texts(
            [query_texts[channel] for channel in channels],
            provider=self.embedding_provider,
            model=self.cache_model,
            embedder=self.embedder,
            batch_size=1,
        )
        return {
            channel: self.indexes[channel].search(
                vector,
                limit=limit,
                filters=filters,
                fetch_multiplier=self.fetch_multiplier,
            )
            for channel, vector in zip(channels, vectors)
        }


class EmbeddingCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_blob BLOB,
                    vector_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(provider, model, text_hash)
                )
                """
            )
            conn.commit()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def embed_texts(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        embedder: Any,
        batch_size: int,
    ) -> list[list[float]]:
        hashes = [SQLiteVecSemanticIndex.text_hash(text) for text in texts]
        cached: dict[str, list[float]] = {}
        with self.connect() as conn:
            for text_hash in sorted(set(hashes)):
                row = conn.execute(
                    """
                    SELECT vector_blob, vector_json
                    FROM embedding_cache
                    WHERE provider = ? AND model = ? AND text_hash = ?
                    """,
                    (provider, model, text_hash),
                ).fetchone()
                if row is not None:
                    cached[text_hash] = decode_vector(row["vector_blob"], row["vector_json"])
        missing_positions = [
            index for index, text_hash in enumerate(hashes) if text_hash not in cached
        ]
        for start in range(0, len(missing_positions), max(1, batch_size)):
            positions = missing_positions[start : start + max(1, batch_size)]
            batch_texts = [texts[index] for index in positions]
            vectors = embed_with_retry(embedder, batch_texts)
            with self.connect() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_cache(
                        provider, model, text_hash, dimension, vector_blob, vector_json
                    )
                    VALUES (?, ?, ?, ?, ?, '')
                    """,
                    [
                        (
                            provider,
                            model,
                            hashes[index],
                            len(vector),
                            encode_vector(vector),
                        )
                        for index, vector in zip(positions, vectors)
                    ],
                )
                conn.commit()
            for index, vector in zip(positions, vectors):
                cached[hashes[index]] = vector
        return [cached[text_hash] for text_hash in hashes]


class OpenAIEmbeddingClient:
    def __init__(self, model: str, *, dimensions: int, timeout: float):
        from openai import OpenAI

        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            timeout=timeout,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions > 0:
            kwargs["dimensions"] = self.dimensions
        response = self.client.embeddings.create(**kwargs)
        return [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]


class HashEmbeddingClient:
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for term in keyword_terms(text)[:256]:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm:
            vector = [value / norm for value in vector]
        return vector


def make_embedder(provider: str, model: str, *, dimensions: int, timeout: float) -> Any:
    if provider == "openai":
        return OpenAIEmbeddingClient(model, dimensions=dimensions, timeout=timeout)
    if provider == "hash":
        return HashEmbeddingClient(dimensions=dimensions if dimensions > 0 else 256)
    raise ValueError(f"unknown embedding provider: {provider}")


def query_text_for_channel(channel: str, query: str, projection: QueryProjection) -> str:
    if channel in {"metadata", "summary"}:
        return query
    if channel == "entity":
        return compact_join(projection.entities, limit=24) or query
    if channel == "constraint":
        return compact_join(projection.constraints, limit=24) or query
    if channel == "relation":
        return "\n".join(projection.relations) or query
    raise ValueError(f"unknown semantic channel: {channel}")


def rank_single_semantic_channel(
    channel: str,
    results: list[SemanticSearchResult],
) -> list[HybridProjectionCandidate]:
    rows: list[HybridProjectionCandidate] = []
    seen: set[str] = set()
    for rank, result in enumerate(results, 1):
        doc_id = str(result.external_id or result.file_ref)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        rows.append(
            HybridProjectionCandidate(
                document_id=doc_id,
                score=1 / (60 + rank),
                sources=[{"channel": channel, "rank": rank, "distance": result.distance}],
                source_type=result.source_type,
                source_path=result.source_path,
                title=result.title,
                metadata=result.metadata,
                snippet=f"{channel}_vector rank={rank}",
            )
        )
    return rows


def aggregate_hybrid_entity_relation(
    channel_hits: dict[str, list[SemanticSearchResult]],
    projection: QueryProjection,
) -> list[HybridProjectionCandidate]:
    by_doc: dict[str, dict[str, Any]] = {}
    for channel, results in channel_hits.items():
        weight = HYBRID_ENTITY_RELATION_WEIGHTS[channel]
        seen_in_channel = set()
        for rank, result in enumerate(results, 1):
            doc_id = str(result.external_id or result.file_ref)
            if doc_id in seen_in_channel:
                continue
            seen_in_channel.add(doc_id)
            item = by_doc.setdefault(
                doc_id,
                {
                    "document_id": doc_id,
                    "score": 0.0,
                    "sources": [],
                    "source_type": result.source_type,
                    "source_path": result.source_path,
                    "title": result.title,
                    "metadata": result.metadata,
                },
            )
            item["score"] += weight * (1 / (60 + rank))
            item["sources"].append({"channel": channel, "rank": rank, "distance": result.distance})
    candidates = []
    for item in by_doc.values():
        item["score"] += exact_match_bonus(item, projection)
        candidates.append(
            HybridProjectionCandidate(
                document_id=item["document_id"],
                score=float(item["score"]),
                sources=item["sources"],
                source_type=item["source_type"],
                source_path=item["source_path"],
                title=item["title"],
                metadata=item["metadata"],
                snippet=hybrid_snippet(item),
            )
        )
    return sorted(
        candidates,
        key=lambda item: (
            -item.score,
            min(source["rank"] for source in item.sources),
            item.document_id,
        ),
    )


def exact_match_bonus(item: dict[str, Any], projection: QueryProjection) -> float:
    haystack = json.dumps(
        {
            "title": item.get("title", ""),
            "source_path": item.get("source_path", ""),
            "metadata": item.get("metadata", {}),
        },
        ensure_ascii=False,
    ).lower()
    terms = [*projection.entities[:8], *projection.constraints[:6]]
    matched = 0
    for term in terms:
        normalized = str(term).lower().strip()
        if len(normalized) >= 3 and normalized in haystack:
            matched += 1
    return min(0.02, matched * 0.004)


def hybrid_snippet(item: dict[str, Any]) -> str:
    channels = ", ".join(
        f"{source['channel']}@{source['rank']}" for source in item.get("sources", [])[:4]
    )
    topic = str((item.get("metadata") or {}).get("topic") or "").strip()
    parts = [f"hybrid_entity_relation_vector {channels}"]
    if topic:
        parts.append(f"topic: {topic}")
    return "; ".join(parts)


def heuristic_query_projection(question: str) -> QueryProjection:
    entities = dedupe(
        [
            *identifier_terms(question),
            *keyword_terms(question)[:16],
        ]
    )[:16]
    constraints = dedupe(
        [
            *extract_constraint_terms(question),
            *numeric_terms(question),
        ]
    )[:12]
    predicate = infer_query_predicate(question)
    subject = entities[0] if entities else "question"
    return QueryProjection(
        entities=entities,
        relations=[f"{subject} | {predicate} | {question}"],
        constraints=constraints,
        expected_answer_type=infer_answer_type(question),
    )


def compact_join(values: list[str], *, limit: int) -> str:
    return " | ".join(values[:limit])


def identifier_terms(text: str) -> list[str]:
    patterns = [
        r"\b[A-Z]{2,12}-\d{2,}\b",
        r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b\s*(?:=|:)\s*[A-Za-z0-9_.:/-]+",
        r"\b[A-Za-z][A-Za-z0-9_+-]+(?:[-_+][A-Za-z0-9]+)+\b",
        r"\b[A-Z]{2,}[A-Za-z0-9_-]*\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(match.strip() for match in re.findall(pattern, text))
    return found


def keyword_terms(text: str) -> list[str]:
    stopwords = {
        "about",
        "after",
        "also",
        "and",
        "are",
        "for",
        "from",
        "how",
        "into",
        "the",
        "this",
        "that",
        "what",
        "when",
        "where",
        "which",
        "with",
    }
    terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_+-]{2,}", text)
        if term.lower() not in stopwords
    ]
    return dedupe(terms)


def extract_constraint_terms(text: str) -> list[str]:
    constraints = []
    for pattern in [
        r"\b(?:must|should|required|requires?|default(?:s)?|limit(?:s)?|maximum|minimum)\b[^.!?\n]{0,120}",
        r"\b[A-Za-z_][A-Za-z0-9_]{2,}\s*(?:=|:)\s*[A-Za-z0-9_.:/-]+",
    ]:
        constraints.extend(match.strip() for match in re.findall(pattern, text, flags=re.IGNORECASE))
    return dedupe(constraints)


def numeric_terms(text: str) -> list[str]:
    return re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:MiB|GiB|MB|GB|ms|sec|seconds|minutes|hours|days|%|tokens?|req/s|rps)\b",
        text,
        flags=re.IGNORECASE,
    )


def infer_query_predicate(question: str) -> str:
    lowered = question.lower()
    rules = [
        ("asks_default", ["default", "defaults"]),
        ("asks_limit", ["limit", "maximum", "minimum", "size"]),
        ("asks_cause", ["caused", "cause", "why"]),
        ("asks_owner", ["who", "owner", "assigned"]),
        ("asks_deadline", ["when", "deadline", "date"]),
        ("asks_status", ["status", "state"]),
        ("asks_requirement", ["required", "requirement", "must"]),
    ]
    for predicate, needles in rules:
        if any(needle in lowered for needle in needles):
            return predicate
    return "asks_about"


def infer_answer_type(question: str) -> str:
    lowered = question.lower()
    if "how many" in lowered or "limit" in lowered or "size" in lowered:
        return "number_or_limit"
    if lowered.startswith("who"):
        return "person_or_team"
    if lowered.startswith("when"):
        return "date_or_time"
    if "why" in lowered or "caused" in lowered:
        return "cause"
    return "fact"


def dedupe(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def embedding_cache_model_key(model: str, dimensions: int) -> str:
    return f"{model}:dimensions={dimensions}" if dimensions > 0 else model


def embed_with_retry(embedder: Any, texts: list[str], *, max_attempts: int = 8) -> list[list[float]]:
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed(texts)
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(min(120.0, 2.0 ** (attempt - 1)))
    raise RuntimeError("unreachable embedding retry state")


def encode_vector(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def decode_vector(blob: bytes | None, vector_json: str | None) -> list[float]:
    if blob:
        if len(blob) % 4 != 0:
            raise ValueError("invalid cached vector blob length")
        return list(struct.unpack(f"<{len(blob) // 4}f", blob))
    if vector_json:
        value = json.loads(vector_json)
        if isinstance(value, list):
            return [float(item) for item in value]
    raise ValueError("cached embedding row does not contain a vector")
