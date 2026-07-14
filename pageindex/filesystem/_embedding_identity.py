from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def normalize_base_url(value: str | None) -> str:
    raw = str(value or DEFAULT_OPENAI_BASE_URL).strip()
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("embedding base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "embedding base URL must not contain credentials, query parameters, or a fragment"
        )
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            "",
            "",
        )
    )


def normalize_model(value: str | None) -> str:
    model = str(value or "").strip()
    if not model:
        raise ValueError("embedding model must not be empty")
    return model
