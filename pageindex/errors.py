class PageIndexAPIError(Exception):
    """status_code carries the HTTP status when the raising site passes it;
    None does not imply local/client-side. rpc_error carries an MCP
    server's JSON-RPC error object as it was sent."""

    def __init__(self, *args: object, status_code: int | None = None,
                 rpc_error: dict | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.rpc_error = rpc_error


class PageIndexTimeoutError(PageIndexAPIError):
    """The whole-run deadline expired; local execution has been torn down.

    Already dispatched remote work may still finish and be billed.
    """

    code = "deadline_exceeded"

    def __init__(self, *args: object, timeout: float | None = None) -> None:
        super().__init__(*args)
        self.timeout = timeout


def _pageindex_cause(exc: BaseException | None) -> PageIndexAPIError | None:
    """The PageIndexAPIError behind a framework's wrapper exception, if any."""
    while exc is not None:
        if isinstance(exc, PageIndexAPIError):
            return exc
        exc = exc.__cause__
    return None
