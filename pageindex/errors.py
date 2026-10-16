class PageIndexAPIError(Exception):
    """status_code carries the HTTP status when the raising site passes it;
    None does not imply local/client-side."""

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


def _pageindex_cause(exc: BaseException | None) -> PageIndexAPIError | None:
    """The PageIndexAPIError behind a framework's wrapper exception, if any."""
    while exc is not None:
        if isinstance(exc, PageIndexAPIError):
            return exc
        exc = exc.__cause__
    return None


class PageIndexTimeoutError(PageIndexAPIError):
    """A chat run stopped at its whole-run ``timeout``. Raised once the
    run is torn down: no further model or tool calls start after it.
    ``code`` is always ``"deadline_exceeded"``; ``timeout`` is the
    budget in seconds."""

    code = "deadline_exceeded"

    def __init__(self, *args: object, timeout: float | None = None) -> None:
        super().__init__(*args)
        self.timeout = timeout
