class PageIndexAPIError(Exception):
    """Custom exception for PageIndex API errors.

    Raised by both cloud mode (HTTP errors from api.pageindex.ai) and local
    mode (missing documents, unusable storage), so code written against the
    0.2.x cloud SDK keeps working unchanged in local mode.
    """
    pass


AUTH_HINT = (
    "api_key must be a PageIndex cloud API key (https://dash.pageindex.ai/api-keys). "
    "For local mode, omit api_key and set your LLM provider key "
    "(e.g. OPENAI_API_KEY) in the environment."
)
