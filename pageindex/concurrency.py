"""
Concurrency throttling for LLM API calls.

Uses a semaphore to limit concurrent LLM requests and avoid HTTP 429 rate limits.
"""
import asyncio


# Default semaphore for throttling concurrent LLM calls
_sem: asyncio.Semaphore | None = None
_max_concurrent: int = 5


def _get_sem() -> asyncio.Semaphore:
    """Get or create the global semaphore instance."""
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(_max_concurrent)
    return _sem


def set_max_concurrent(max_concurrent: int) -> None:
    """Set the maximum number of concurrent LLM calls."""
    global _max_concurrent, _sem
    _max_concurrent = max_concurrent
    # Reset semaphore so it gets recreated with new limit on next call
    _sem = None


def get_max_concurrent() -> int:
    """Get the current max concurrent setting."""
    return _max_concurrent


async def limited_llm_acompletion(model, prompt):
    """
    Wrapper around llm_acompletion that limits concurrent calls via semaphore.
    """
    # Import here to avoid circular import
    from .utils import llm_acompletion
    sem = _get_sem()
    async with sem:
        return await llm_acompletion(model, prompt)
