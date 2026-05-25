from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol


GENERATED_METADATA_FIELDS = ("summary", "doc_type", "domain", "topic", "entity", "relation")


class MetadataGenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class MetadataGenerationInput:
    file_ref: str
    external_id: str | None
    title: str
    source_path: str
    content_type: str
    source_type: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    text_artifact_path: str | None = None


@dataclass(frozen=True)
class MetadataGenerationResult:
    values: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


class MetadataGenerator(Protocol):
    def generate(
        self,
        request: MetadataGenerationInput,
        *,
        fields: list[str],
    ) -> MetadataGenerationResult | dict[str, Any]:
        ...


class OpenAIMetadataGenerator:
    """Default product generator for retrieval metadata.

    This intentionally lives under pageindex.filesystem instead of benchmark
    paths. It uses registered text today; callers can pass PageIndex-extracted
    text through the same MetadataGenerationInput without changing the API.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        max_text_chars: int = 24000,
    ):
        self.model = model or os.environ.get("PIFS_METADATA_MODEL", "gpt-5-nano")
        self.base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
        self.max_text_chars = max_text_chars

    def generate(
        self,
        request: MetadataGenerationInput,
        *,
        fields: list[str],
    ) -> MetadataGenerationResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise MetadataGenerationError("OPENAI_API_KEY is required for PIFS metadata generation")

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate grounded retrieval metadata for one document. "
                        "Use only the provided document text and ordinary source metadata. "
                        "The summary must be a retrieval summary, not a title rewrite. "
                        "Do not use filenames, paths, URLs, storage URIs, or outside knowledge. "
                        "Return strict JSON matching the requested fields."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "requested_fields": fields,
                            "document": {
                                "title": request.title,
                                "source_type": request.source_type,
                                "content_type": request.content_type,
                                "metadata": request.metadata,
                                "text": request.text[: self.max_text_chars],
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format=self._response_format(fields),
        )
        content = response.choices[0].message.content or "{}"
        values = json.loads(content)
        return MetadataGenerationResult(
            values={field: values[field] for field in fields if field in values},
        )

    @staticmethod
    def _response_format(fields: list[str]) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for field in fields:
            if field in {"summary", "doc_type", "domain", "topic"}:
                properties[field] = {"type": "string"}
            elif field in {"entity", "relation"}:
                properties[field] = {"type": "string"}
            else:
                raise MetadataGenerationError(
                    f"OpenAIMetadataGenerator does not support generated metadata field: {field}"
                )
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "pifs_metadata_generation",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": fields,
                    "properties": properties,
                },
            },
        }
