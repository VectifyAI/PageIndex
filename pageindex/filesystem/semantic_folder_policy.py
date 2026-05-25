from __future__ import annotations

import re
from typing import Any, Iterable


SEMANTIC_FOLDER_ROOT = "/semantic"
SEMANTIC_FOLDER_BASE_FIELDS = {"doc_type", "domain", "topic"}
SEMANTIC_FOLDER_SYSTEM_FIELDS = {"source_type"}
SEMANTIC_FOLDER_FORBIDDEN_FIELDS = {
    "summary",
    "entities",
    "relations",
    "constraints",
    "retrieval_cues",
    "dataset_doc_uuid",
    "path",
    "uri",
    "source_path",
    "storage_uri",
    "title",
    "content_type",
    "created_at",
    "updated_at",
}


def canonical_semantic_folder_field_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").casefold()


def compact_semantic_folder_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", canonical_semantic_folder_field_name(value))


def semantic_folder_field_identity_keys(value: Any) -> frozenset[str]:
    canonical = canonical_semantic_folder_field_name(value)
    compact = compact_semantic_folder_field_name(value)
    return frozenset(key for key in (canonical, compact) if key)


def semantic_folder_field_identity_set(fields: Iterable[Any]) -> frozenset[str]:
    keys: set[str] = set()
    for field in fields:
        keys.update(semantic_folder_field_identity_keys(field))
    return frozenset(keys)


SEMANTIC_FOLDER_FORBIDDEN_FIELD_IDENTITIES = semantic_folder_field_identity_set(
    SEMANTIC_FOLDER_FORBIDDEN_FIELDS
)


def is_semantic_folder_forbidden_field(value: Any) -> bool:
    return bool(
        semantic_folder_field_identity_keys(value)
        & SEMANTIC_FOLDER_FORBIDDEN_FIELD_IDENTITIES
    )


def semantic_folder_allowed_extension_fields(fields: Iterable[Any]) -> set[str]:
    allowed = set()
    for field in fields:
        name = canonical_semantic_folder_field_name(field)
        if name and not is_semantic_folder_forbidden_field(field):
            allowed.add(name)
    return allowed
