"""One retry rule for every model and cloud call the SDK sends itself, after the official OpenAI and Anthropic SDKs."""
import random

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RETRY_STATUS = frozenset({408, 409, 429})

# Retries after the first try, and the longest backoff wait in seconds; reassign to change.
INDEX_RETRIES, INDEX_MAX_DELAY = 9, 30
SDK_RETRIES, SDK_MAX_DELAY = 2, 8


def should_retry(exc):
    """A LiteLLM error worth another try: a network failure or a server's 408/409/429/5xx (x-should-retry overrides), never a local one."""
    import httpx
    chain, link = [], exc
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__cause__ or link.__context__
    if any(isinstance(link, httpx.TransportError)
           and not isinstance(link, (httpx.LocalProtocolError, httpx.UnsupportedProtocol)) for link in chain):
        return True
    if not any(isinstance(link, httpx.HTTPStatusError) for link in chain):
        return False
    verdict = (getattr(exc, "litellm_response_headers", None) or {}).get("x-should-retry")
    if verdict in ("true", "false"):
        return verdict == "true"
    status = getattr(exc, "status_code", None) or 0
    return status in RETRY_STATUS or status >= 500


def retry_after(headers):
    """The server's Retry-After in seconds (retry-after-ms first) when it names 0-60, else None."""
    for key, per_second in (("retry-after-ms", 1000), ("retry-after", 1)):
        try:
            seconds = float((headers or {}).get(key, "")) / per_second
        except (TypeError, ValueError):
            continue
        return seconds if 0 < seconds <= 60 else None
    return None


def delay(attempt, headers, cap):
    """Seconds before retry `attempt` (0-based): the server's Retry-After, else doubling up to `cap`, jittered."""
    return retry_after(headers) or min(cap, 2 ** attempt) * random.uniform(0.5, 1)


def agent_policy(context):
    """openai-agents' runner hook: the rule above on the LiteLLM chat lane."""
    from agents.retry import RetryDecision
    if not should_retry(context.error):
        return False
    headers = getattr(context.error, "litellm_response_headers", None)
    return RetryDecision(retry=True, delay=delay(context.attempt - 1, headers, SDK_MAX_DELAY))


class _CloudRetry(Retry):
    def is_retry(self, method, status_code, has_retry_after=False):
        # A POST carries no idempotency key yet: resent only when refused, as nothing ran.
        if method.upper() == "POST":
            return status_code == 429
        return super().is_retry(method, status_code, has_retry_after)

    def get_retry_after(self, response):
        return retry_after(response.headers)

    def get_backoff_time(self):
        return delay(len(self.history) - 1, None, SDK_MAX_DELAY)


def cloud_session():
    """Cloud API session: GET/PUT/DELETE retry; a POST only if it was never sent or got a 429."""
    retry = _CloudRetry(total=SDK_RETRIES, other=0, status_forcelist=(*RETRY_STATUS, *range(500, 600)),
                        raise_on_status=False)
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
