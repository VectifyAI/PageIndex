class PageIndexAPIError(Exception):
    """status_code carries the HTTP status when the raising site passes it
    (some cloud paths raise bare), so None does not imply local/client-side."""

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code
