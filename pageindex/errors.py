from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .usage import ChatUsage


class PageIndexAPIError(Exception):
    """status_code carries the HTTP status when the raising site passes it;
    None does not imply local/client-side. usage carries the measurement
    of an own-model chat run that failed (``pageindex.ChatUsage``), the
    calls it already made included; None elsewhere."""

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.usage: ChatUsage | None = None


def _pageindex_cause(exc: BaseException | None) -> PageIndexAPIError | None:
    """The PageIndexAPIError behind a framework's wrapper exception, if any."""
    while exc is not None:
        if isinstance(exc, PageIndexAPIError):
            return exc
        exc = exc.__cause__
    return None
