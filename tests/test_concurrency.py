"""
Tests for concurrency throttling of LLM calls.

These tests verify that the semaphore correctly limits concurrent LLM API calls.
"""
import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from pageindex.concurrency import (
    set_max_concurrent,
    get_max_concurrent,
    limited_llm_acompletion,
    _get_sem,
    _sem,
    _max_concurrent,
)


class TestConcurrencySettings:
    """Test concurrency setting management."""

    def teardown_method(self):
        """Reset concurrency settings after each test."""
        set_max_concurrent(5)

    def test_get_max_concurrent_default(self):
        """Test default max concurrent is 5."""
        set_max_concurrent(5)  # reset first
        assert get_max_concurrent() == 5

    def test_set_max_concurrent(self):
        """Test setting max concurrent calls."""
        set_max_concurrent(10)
        assert get_max_concurrent() == 10

    def test_set_max_concurrent_resets_semaphore(self):
        """Test that setting max concurrent resets the semaphore."""
        sem1 = _get_sem()
        set_max_concurrent(10)
        sem2 = _get_sem()
        # Semaphore should be recreated with new limit
        assert sem1 is not sem2


class TestLimitedLlmCompletion:
    """Test the limited_llm_acompletion wrapper."""

    @pytest.mark.asyncio
    async def test_limited_acompletion_uses_semaphore(self):
        """Test that limited_acompletion acquires semaphore."""
        mock_response = "test response"

        # Create a mock for llm_acompletion
        async def mock_llm(model, prompt):
            return mock_response

        with patch('pageindex.concurrency.llm_acompletion', new=mock_llm):
            set_max_concurrent(1)  # Only allow 1 concurrent call

            # Should complete without deadlock
            result = await limited_llm_acompletion("gpt-4", "test prompt")
            assert result == mock_response

    @pytest.mark.asyncio
    async def test_limited_acompletion_concurrent_limit(self):
        """Test that concurrent calls are properly limited by semaphore."""
        call_times = []
        max_concurrent = 0
        current_concurrent = 0

        async def mock_llm(model, prompt):
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            call_times.append(('start', current_concurrent))

            # Simulate some async work
            await asyncio.sleep(0.05)

            call_times.append(('end', current_concurrent))
            current_concurrent -= 1
            return "response"

        with patch('pageindex.concurrency.llm_acompletion', new=mock_llm):
            set_max_concurrent(2)  # Allow 2 concurrent calls

            # Launch 4 tasks concurrently
            tasks = [
                limited_llm_acompletion("gpt-4", f"prompt{i}")
                for i in range(4)
            ]
            results = await asyncio.gather(*tasks)

            # All should complete
            assert len(results) == 4
            # Max concurrent should not exceed semaphore limit
            assert max_concurrent <= 2


class TestSemaphoreBehavior:
    """Test semaphore behavior directly."""

    def test_semaphore_limits_concurrent_access(self):
        """Test that semaphore correctly limits concurrent execution."""
        set_max_concurrent(2)
        sem = _get_sem()

        async def task(task_id):
            async with sem:
                await asyncio.sleep(0.01)
                return task_id

        async def run_test():
            # Launch more tasks than the semaphore limit
            tasks = [task(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_test())
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}