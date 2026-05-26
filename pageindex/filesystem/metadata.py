from __future__ import annotations

import json
import re
from typing import Any

from .types import MetadataField


class MetadataQueryError(ValueError):
    pass


class MetadataQueryEngine:
    FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
    OPERATORS = {"$eq", "$ne", "$in", "$gt", "$gte", "$lt", "$lte", "$contains"}
    LOGICAL_OPERATORS = {"$and", "$or"}
    FOLDER_SCOPE_FIELD_HINTS = {"path", "folder", "folders", "folder_path", "folder_paths"}
    MAX_DEPTH = 5

    def __init__(self, store: Any):
        self.store = store

    def register_schema(self, schema: dict[str, Any], source: str = "manual") -> None:
        fields = []
        raw_fields = schema.get("fields", schema)
        if not isinstance(raw_fields, dict):
            raise MetadataQueryError("metadata schema must contain a fields object")
        for name, declaration in raw_fields.items():
            name = str(name)
            self.validate_field_name(name)
            if isinstance(declaration, str):
                field_type = declaration
                description = ""
            elif isinstance(declaration, dict):
                field_type = str(declaration.get("type", ""))
                description = str(declaration.get("description", ""))
            else:
                raise MetadataQueryError(f"Invalid schema declaration for field: {name}")
            if field_type not in {"string", "number", "boolean"}:
                raise MetadataQueryError(f"Unsupported metadata field type for {name}: {field_type}")
            fields.append(
                MetadataField(
                    name=name,
                    field_type=field_type,
                    description=description,
                    source=source,
                )
            )
        if fields:
            self.store.upsert_metadata_fields(fields)

    def parse_filter(self, value: str | dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            value = self.parse_dsl(value)
        if not isinstance(value, dict):
            raise MetadataQueryError("metadata_filter must be a JSON object")
        self.validate_filter(value)
        return value

    def parse_dsl(self, dsl: str) -> dict[str, Any]:
        try:
            parsed = json.loads(dsl)
        except json.JSONDecodeError as exc:
            raise MetadataQueryError(
                "metadata DSL must be a JSON object, for example "
                '\'{"$and":[{"repo":"redwood"},{"year":{"$gte":2024}}]}\''
            ) from exc
        if not isinstance(parsed, dict):
            raise MetadataQueryError("metadata DSL must be a JSON object")
        return parsed

    def validate_filter(self, metadata_filter: dict[str, Any], depth: int = 1) -> None:
        if depth > self.MAX_DEPTH:
            raise MetadataQueryError(f"metadata_filter nesting depth exceeds {self.MAX_DEPTH}")
        if not metadata_filter:
            return
        for key, condition in metadata_filter.items():
            if key in self.LOGICAL_OPERATORS:
                self._validate_logical(key, condition, depth)
                continue
            self.validate_field(key)
            self._validate_field_condition(key, condition)

    def _validate_logical(self, operator: str, condition: Any, depth: int) -> None:
        if not isinstance(condition, list) or not condition:
            raise MetadataQueryError(f"{operator} requires a non-empty list")
        for item in condition:
            if not isinstance(item, dict):
                raise MetadataQueryError(f"{operator} items must be metadata filter objects")
            self.validate_filter(item, depth + 1)

    def _validate_field_condition(self, field: str, condition: Any) -> None:
        if not isinstance(condition, dict) or not any(
            str(key).startswith("$") for key in condition
        ):
            self._validate_scalar(condition, context=field)
            return
        if len(condition) != 1:
            raise MetadataQueryError(
                f"Field {field} condition must contain exactly one metadata operator"
            )
        operator, expected = next(iter(condition.items()))
        if operator not in self.OPERATORS:
            raise MetadataQueryError(f"Unsupported metadata operator: {operator}")
        if operator == "$in":
            if not isinstance(expected, list):
                raise MetadataQueryError(f"{field} $in requires a list")
            for item in expected:
                self._validate_scalar(item, context=f"{field} $in")
            return
        if operator == "$contains":
            self._validate_scalar(expected, context=f"{field} $contains")
            return
        if operator in {"$gt", "$gte", "$lt", "$lte"}:
            self._validate_range_value(expected, context=f"{field} {operator}")
            return
        self._validate_scalar(expected, context=f"{field} {operator}")

    def validate_field(self, field: str) -> None:
        self.validate_field_name(field)
        if not self.store.metadata_field_exists(field):
            if field in self.FOLDER_SCOPE_FIELD_HINTS:
                raise MetadataQueryError(
                    f"Unknown metadata field: {field}. Folder paths are positional PIFS paths, "
                    "not metadata fields; use `ls /documents` or `find /documents -type f`. "
                    "Use --where only with fields from `stat --schema`."
                )
            raise MetadataQueryError(f"Unknown metadata field: {field}")

    def validate_field_name(self, field: str) -> None:
        if not self.FIELD_RE.match(field):
            raise MetadataQueryError(f"Invalid metadata field: {field}")

    def export_schema(self) -> dict[str, Any]:
        fields = {}
        for field in self.store.list_metadata_fields():
            fields[field.name] = {
                "type": field.field_type,
                "description": field.description,
            }
        return {"fields": fields}

    @staticmethod
    def _validate_scalar(value: Any, *, context: str) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            return
        if isinstance(value, str):
            return
        raise MetadataQueryError(f"{context} must be a string, number, or boolean")

    @staticmethod
    def _validate_range_value(value: Any, *, context: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise MetadataQueryError(f"{context} must be a string or number")
