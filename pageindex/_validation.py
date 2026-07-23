from __future__ import annotations

from .errors import PageIndexError


MAX_COLLECTION_NAME_LENGTH = 255


def validate_collection_name(name: str) -> None:
    """Validate the collection-name contract shared by local and cloud modes.

    Collection names are logical identifiers, not filesystem paths.  Keep this
    in sync with the cloud folder API, which accepts any non-empty Unicode
    string up to 255 characters.
    """
    invalid = (
        not isinstance(name, str)
        or not name
        or len(name) > MAX_COLLECTION_NAME_LENGTH
    )
    if not invalid:
        try:
            name.encode("utf-8")
        except UnicodeEncodeError:
            # JSON/database boundaries require Unicode scalar values; lone UTF-16
            # surrogates are Python strings but cannot be encoded as valid UTF-8.
            invalid = True
    if invalid:
        raise PageIndexError(
            f"Invalid collection name: {name!r}. "
            f"Must be a non-empty string of valid Unicode with at most "
            f"{MAX_COLLECTION_NAME_LENGTH} characters."
        )
