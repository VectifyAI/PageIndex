class PageIndexAPIError(Exception):
    """Raised when the SDK reaches a provider or the agent backend and
    the call fails (transport error, model returned no answer, tool
    runner misbehaved). ``status_code`` carries the HTTP status when the
    raising site passes it; a missing ``status_code`` is consistent with
    a non-HTTP execution failure and does not imply local validation."""

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code


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
