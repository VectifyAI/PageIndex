"""PageIndex naming v1. Keep the shared naming-v1.json contract in sync."""

import hashlib
import os
import re
import unicodedata

MAX_NAME_BYTES = 180
_CONTROLS = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_ILLEGAL = re.compile(r'[<>:"/\\|?*]')
_SURROGATES = re.compile(r"[\ud800-\udfff]")
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?$", re.I)
# Use the same whitespace set as JavaScript, rather than Python's broader \s.
_WHITESPACE = re.compile(
    r"[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+"
)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'", "“": '"', "”": '"'})


def normalize_filename(name: str) -> str:
    """Normalize Unicode, quote variants and whitespace; preserve case."""
    return _WHITESPACE.sub(
        " ", unicodedata.normalize("NFKC", name).translate(_QUOTES)
    ).strip(" ")


def _prefix(value: str, max_bytes: int) -> str:
    return value.encode("utf-8")[: max(0, max_bytes)].decode("utf-8", errors="ignore")


def truncate_filename(
    name: str, max_bytes: int = MAX_NAME_BYTES, *, suffix: str = ""
) -> str:
    """Fit a name including its extension and collision suffix into the byte budget."""
    base, ext = os.path.splitext(name)
    candidate = base + suffix + ext
    if len(candidate.encode("utf-8")) <= max_bytes:
        return candidate
    digest = hashlib.md5(name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    ending = "_" + digest + suffix
    budget = max_bytes - len(ending.encode("utf-8"))
    if budget < 4:
        raise ValueError("Filename byte limit is too small for its suffix")
    # An unbounded extension must not defeat the total limit. Reserve one
    # complete character of the basename even in that case.
    first_bytes = len((base[:1] or "_").encode("utf-8"))
    ext = _prefix(ext, budget - first_bytes).rstrip(" .")
    base = _prefix(base, budget - len(ext.encode("utf-8"))) or "_"
    return base + ending + ext


def sanitize_filename(name: str, max_bytes: int = MAX_NAME_BYTES) -> str:
    """Choose a legal upload name. Only call before assigning its storage key."""
    name = _CONTROLS.sub("_", _SURROGATES.sub("\ufffd", name))
    name = _ILLEGAL.sub("_", normalize_filename(name)).rstrip(" .") or "untitled"
    if _RESERVED.fullmatch(name):
        name = "_" + name
    return truncate_filename(name, max_bytes)


def validate_folder_name(name: str) -> str:
    """Normalize a folder name, rejecting characters that would need replacement."""
    if not isinstance(name, str):
        raise ValueError("Folder name must be a string")
    if _CONTROLS.search(name) or _SURROGATES.search(name):
        raise ValueError(
            "Folder name cannot contain control characters or invalid Unicode"
        )
    name = normalize_filename(name)
    if not name:
        raise ValueError("Folder name cannot be empty")
    if _ILLEGAL.search(name):
        raise ValueError('Folder name cannot contain / \\ : * ? " < > |')
    if name in {".", ".."} or _RESERVED.fullmatch(name):
        raise ValueError("Folder name is reserved")
    if name.endswith("."):
        raise ValueError("Folder name cannot end with a period")
    if len(name.encode("utf-8")) > MAX_NAME_BYTES:
        raise ValueError("Folder name must be at most 180 UTF-8 bytes")
    return name


def validate_stored_filename(name: str) -> str:
    """Read an already assigned name literally, including names predating v1."""
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or _CONTROLS.search(name)
        or _SURROGATES.search(name)
    ):
        raise ValueError(
            "File name must be a single path component without control characters"
        )
    return name
