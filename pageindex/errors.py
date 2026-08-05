class PageIndexAPIError(Exception):
    """Custom exception for PageIndex API errors.

    Raised by both cloud mode (HTTP errors from api.pageindex.ai) and local
    mode (missing documents, unusable storage), so code written against the
    0.2.x cloud SDK keeps working unchanged in local mode.
    """
    pass
