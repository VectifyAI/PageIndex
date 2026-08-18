class PageIndexAPIError(Exception):
    """status_code carries the HTTP status when the failure came from a
    non-200 cloud response; None otherwise (local mode, client-side)."""

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code
