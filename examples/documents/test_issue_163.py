import pytest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pageindex.page_index import (
    check_if_toc_extraction_is_complete,
    check_if_toc_transformation_is_complete,
    toc_detector_single_page,
    detect_page_index,
    extract_toc_content,
    toc_transformer,
)


class TestRobustKeyAccess:
    @patch("pageindex.page_index.llm_completion", return_value="")
    def test_toc_detector_empty_response(self, mock_llm):
        result = toc_detector_single_page("some content", model="test")
        assert result == "no"

    @patch("pageindex.page_index.llm_completion", return_value='{"toc_detected": "yes"}')
    def test_toc_detector_valid_response(self, mock_llm):
        result = toc_detector_single_page("some content", model="test")
        assert result == "yes"

    @patch("pageindex.page_index.llm_completion", return_value="not json at all")
    def test_toc_detector_malformed_response(self, mock_llm):
        result = toc_detector_single_page("some content", model="test")
        assert result == "no"

    @patch("pageindex.page_index.llm_completion", return_value="")
    def test_extraction_complete_empty_response(self, mock_llm):
        result = check_if_toc_extraction_is_complete("doc", "toc", model="test")
        assert result == "no"

    @patch("pageindex.page_index.llm_completion", return_value='{"completed": "yes"}')
    def test_extraction_complete_valid_response(self, mock_llm):
        result = check_if_toc_extraction_is_complete("doc", "toc", model="test")
        assert result == "yes"

    @patch("pageindex.page_index.llm_completion", return_value="")
    def test_transformation_complete_empty_response(self, mock_llm):
        result = check_if_toc_transformation_is_complete("raw", "cleaned", model="test")
        assert result == "no"

    @patch("pageindex.page_index.llm_completion", return_value='{"thinking": "looks fine", "completed": "yes"}')
    def test_transformation_complete_valid_response(self, mock_llm):
        result = check_if_toc_transformation_is_complete("raw", "cleaned", model="test")
        assert result == "yes"

    @patch("pageindex.page_index.llm_completion", return_value="")
    def test_detect_page_index_empty_response(self, mock_llm):
        result = detect_page_index("toc text", model="test")
        assert result == "no"


class TestExtractTocContentRetryLoop:
    @patch("pageindex.page_index.check_if_toc_transformation_is_complete")
    @patch("pageindex.page_index.llm_completion")
    def test_completes_on_first_try(self, mock_llm, mock_check):
        mock_llm.return_value = ("full toc content", "finished")
        mock_check.return_value = "yes"
        result = extract_toc_content("raw content", model="test")
        assert result == "full toc content"
        assert mock_llm.call_count == 1

    @patch("pageindex.page_index.check_if_toc_transformation_is_complete")
    @patch("pageindex.page_index.llm_completion")
    def test_continues_on_incomplete(self, mock_llm, mock_check):
        mock_llm.side_effect = [
            ("partial toc", "max_output_reached"),
            (" continued toc", "finished"),
        ]
        mock_check.side_effect = ["no", "yes"]
        result = extract_toc_content("raw content", model="test")
        assert result == "partial toc continued toc"
        assert mock_llm.call_count == 2

    @patch("pageindex.page_index.check_if_toc_transformation_is_complete")
    @patch("pageindex.page_index.llm_completion")
    def test_max_retries_raises_exception(self, mock_llm, mock_check):
        mock_llm.return_value = ("chunk", "max_output_reached")
        mock_check.return_value = "no"
        with pytest.raises(Exception, match="Failed to complete table of contents extraction"):
            extract_toc_content("raw content", model="test")
        assert mock_llm.call_count == 6

    @patch("pageindex.page_index.check_if_toc_transformation_is_complete")
    @patch("pageindex.page_index.llm_completion")
    def test_chat_history_grows_incrementally(self, mock_llm, mock_check):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ("initial", "max_output_reached")
            if call_count[0] == 2:
                history = kwargs.get("chat_history", [])
                assert len(history) == 2
                return (" part2", "max_output_reached")
            if call_count[0] == 3:
                history = kwargs.get("chat_history", [])
                assert len(history) == 4
                return (" part3", "finished")
            return ("", "finished")

        mock_llm.side_effect = side_effect
        mock_check.side_effect = ["no", "no", "yes"]
        result = extract_toc_content("raw content", model="test")
        assert result == "initial part2 part3"


class TestTocTransformerRetryLoop:
    @patch("pageindex.page_index.check_if_toc_transformation_is_complete")
    @patch("pageindex.page_index.llm_completion")
    def test_completes_on_first_try(self, mock_llm, mock_check):
        mock_llm.return_value = (
            '{"table_of_contents": [{"structure": "1", "title": "Intro", "page": 1}]}',
            "finished",
        )
        mock_check.return_value = "yes"
        result = toc_transformer("raw toc", model="test")
        assert len(result) == 1
        assert result[0]["title"] == "Intro"

    @patch("pageindex.page_index.check_if_toc_transformation_is_complete")
    @patch("pageindex.page_index.llm_completion")
    def test_handles_missing_table_of_contents_key(self, mock_llm, mock_check):
        mock_llm.return_value = ('{"other_key": "value"}', "finished")
        mock_check.return_value = "yes"
        result = toc_transformer("raw toc", model="test")
        assert result == []
