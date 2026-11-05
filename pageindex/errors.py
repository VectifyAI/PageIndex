class PageIndexAPIError(Exception):
    """status_code carries the HTTP status when the raising site passes it;
    None does not imply local/client-side. rpc_error carries an MCP
    server's JSON-RPC error object as it was sent. usage is set when an
    own-model chat run fails: what its finished model calls had spent, in
    the protocol's usage shape with each call listed (claude-code/: the
    run's total only, None on a tool failure or if the run ends before
    Claude Code reports it); None on any other error. partial_answer is
    the text a non-streamed chat() answer, or a Messages stream's
    get_final_text(), reached before it was cut short (a stream has already
    delivered its text); None otherwise."""

    def __init__(self, *args: object, status_code: int | None = None,
                 rpc_error: dict | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code
        self.rpc_error = rpc_error
        self.usage: dict | None = None
        self.partial_answer: str | None = None


class PageIndexValidationError(PageIndexAPIError):
    """The SDK refused the call before reaching a provider or the agent
    backend: an argument had the wrong shape, a managed field was
    passed where the SDK owns the value, the requested protocol needs
    a missing dependency, or the configured model cannot drive the
    chosen lane. ``status_code`` is always ``None`` — there was no
    request to time out or to receive a status from.

    Use ``except PageIndexValidationError`` (and ``except
    PageIndexAPIError`` for the rest) to tell caller-side mistakes apart
    from upstream failures without inspecting exception messages."""


def _pageindex_cause(exc: BaseException | None) -> PageIndexAPIError | None:
    """The PageIndexAPIError behind a framework's wrapper exception, if any."""
    while exc is not None:
        if isinstance(exc, PageIndexAPIError):
            return exc
        exc = exc.__cause__
    return None
