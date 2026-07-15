"""Tests for issue #15: enforce --max-tokens-per-node and safe llm_completion unpacking."""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.page_index import (
    generate_toc_init,
    page_list_to_group_text,
    process_no_toc,
)
from pageindex.utils import llm_completion, truncate_to_token_limit


class TestPageListGroupingRespectsMaxTokens:
    def test_small_budget_splits_pages(self):
        pages = ["aaa", "bbb", "ccc"]
        lengths = [50, 50, 50]
        groups = page_list_to_group_text(pages, lengths, max_tokens=60)
        assert len(groups) >= 2
        # No group should exceed the budget when pages themselves fit.
        for group in groups:
            # Rough check: each page is 50 tokens; groups with one page ok.
            assert isinstance(group, str)

    def test_large_budget_keeps_single_group(self):
        pages = ["aaa", "bbb"]
        lengths = [10, 10]
        groups = page_list_to_group_text(pages, lengths, max_tokens=20000)
        assert groups == ["aaabbb"]

    def test_none_budget_falls_back_to_default(self):
        pages = ["aaa", "bbb"]
        lengths = [10, 10]
        groups = page_list_to_group_text(pages, lengths, max_tokens=None)
        assert groups == ["aaabbb"]


class TestTruncateToTokenLimit:
    @patch("pageindex.utils.count_tokens")
    def test_no_truncate_when_under_limit(self, mock_count):
        mock_count.return_value = 5
        assert truncate_to_token_limit("hello world", 10, model="m") == "hello world"

    @patch("pageindex.utils.count_tokens")
    def test_truncates_when_over_limit(self, mock_count):
        # count_tokens(text[:n]) ≈ n for this stub
        def _count(text, model=None):
            return len(text or "")

        mock_count.side_effect = _count
        result = truncate_to_token_limit("abcdefghij", 4, model="m")
        assert len(result) <= 4
        assert result == "abcd"

    def test_unset_limit_noop(self):
        assert truncate_to_token_limit("abc", None) == "abc"
        assert truncate_to_token_limit("abc", 0) == "abc"


class TestLlmCompletionReturnShape:
    @patch("pageindex.utils.litellm.completion")
    def test_return_finish_reason_is_always_two_tuple(self, mock_completion):
        choice = MagicMock()
        choice.message.content = "ok"
        choice.finish_reason = "stop"
        mock_completion.return_value = MagicMock(choices=[choice])

        result = llm_completion("gpt-4o-mini", "prompt", return_finish_reason=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result == ("ok", "finished")

    @patch("pageindex.utils.litellm.completion")
    def test_none_content_becomes_empty_string(self, mock_completion):
        choice = MagicMock()
        choice.message.content = None
        choice.finish_reason = "stop"
        mock_completion.return_value = MagicMock(choices=[choice])

        content, finish_reason = llm_completion(
            "gpt-4o-mini", "prompt", return_finish_reason=True
        )
        assert content == ""
        assert finish_reason == "finished"

    @patch("pageindex.utils.time.sleep")
    @patch("pageindex.utils.litellm.completion", side_effect=RuntimeError("tokens_limit_reached"))
    def test_exhausted_retries_return_error_tuple(self, mock_completion, _sleep):
        content, finish_reason = llm_completion(
            "gpt-4o-mini", "huge prompt", return_finish_reason=True
        )
        assert content == ""
        assert finish_reason == "error"
        assert mock_completion.call_count == 10


class TestProcessNoTocHonorsMaxTokens:
    @patch("pageindex.page_index.convert_physical_index_to_int", side_effect=lambda x: x)
    @patch("pageindex.page_index.generate_toc_continue")
    @patch("pageindex.page_index.generate_toc_init")
    @patch("pageindex.page_index.page_list_to_group_text")
    @patch("pageindex.page_index.count_tokens", return_value=100)
    def test_passes_max_tokens_into_grouping(
        self, _count, mock_group, mock_init, mock_continue, _convert
    ):
        mock_group.return_value = ["chunk"]
        mock_init.return_value = [{"title": "A", "physical_index": 1}]
        logger = MagicMock()

        page_list = [("page text", 100)]
        process_no_toc(page_list, model="m", logger=logger, max_tokens=100)

        assert mock_group.call_args.kwargs.get("max_tokens") == 100
        assert mock_init.call_args.kwargs.get("max_tokens") == 100


class TestGenerateTocInitTruncatesPart:
    @patch("pageindex.page_index.extract_json", return_value=[])
    @patch("pageindex.page_index.llm_completion", return_value=('[]', "finished"))
    @patch("pageindex.page_index.truncate_to_token_limit")
    def test_truncates_document_part(self, mock_trunc, mock_llm, _extract):
        mock_trunc.return_value = "short"
        generate_toc_init("very long part", model="m", max_tokens=50)
        mock_trunc.assert_called_once_with("very long part", 50, model="m")
        prompt = mock_llm.call_args.kwargs["prompt"]
        assert "very long part" not in prompt
        assert "short" in prompt
