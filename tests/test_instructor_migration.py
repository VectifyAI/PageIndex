"""
Tests for the Instructor + Pydantic structured-output refactor.

These replace tests/test_issue_163.py, which asserted implementation details of
the old manual-JSON-parsing code (dict .get() fallbacks, chat-history-growing
continuation loops, retry counts tied to "max_output_reached" finish_reason
strings). That code path no longer exists after this refactor — validation,
retries, and truncation handling are now owned by Instructor/Pydantic — so
those tests were superseded rather than adapted.

Adjust import paths below to match your actual module layout if they differ.
"""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, ValidationError
from typing import Optional, List

from pageindex.schemas import (
    TOCIndexItem,
    TOCIndexList,
    TOCTransformation,
    TOCItem,
)
from pageindex.page_index import (
    toc_transformer,
    add_page_offset_to_toc_json,
)
from pageindex.utils import llm_structured


# ---------------------------------------------------------------------------
# Schema-level tests — catch the exact bug class this PR fixed
# (Optional[str] without default=None is NOT omittable in Pydantic v2)
# ---------------------------------------------------------------------------

class TestSchemaOptionalFields:
    def test_toc_index_item_structure_is_omittable(self):
        """
        Regression test for the bug found during testing: TOCIndexItem.structure
        was declared Optional[str] with no default, which does NOT make the
        field omittable in Pydantic v2 — any model response omitting the key
        failed with "Field required" regardless of which LLM produced it.
        """
        item = TOCIndexItem(title="Some Section")  # structure omitted entirely
        assert item.structure is None
        assert item.physical_index is None

    def test_toc_index_item_accepts_explicit_none(self):
        item = TOCIndexItem(title="Some Section", structure=None, physical_index=None)
        assert item.structure is None

    def test_toc_index_item_requires_title(self):
        with pytest.raises(ValidationError):
            TOCIndexItem(structure="1.1")  # title is required, correctly

    def test_toc_index_item_physical_index_pattern_enforced(self):
        with pytest.raises(ValidationError):
            TOCIndexItem(title="X", physical_index="page 5")  # wrong format

        item = TOCIndexItem(title="X", physical_index="<physical_index_5>")
        assert item.physical_index == "<physical_index_5>"

    def test_toc_index_list_requires_items_key(self):
        with pytest.raises(ValidationError):
            TOCIndexList()  # items is required — this is correct/intended,
            # unlike `structure` above which should be optional


# ---------------------------------------------------------------------------
# add_page_offset_to_toc_json — regression test for the None-offset crash
# ---------------------------------------------------------------------------

class TestAddPageOffsetToTocJson:
    def test_raises_clear_error_when_offset_is_none(self):
        """
        Regression test: calculate_page_offset can return None when zero TOC
        entries were matched to a physical_index. The unguarded code evaluated
        int + None, raising an opaque TypeError. This should now raise a
        descriptive ValueError instead.
        """
        toc_data = [{"title": "Intro", "page": 1, "structure": "1"}]
        with pytest.raises(ValueError, match="page offset"):
            add_page_offset_to_toc_json(toc_data, offset=None)

    def test_applies_offset_correctly_when_valid(self):
        toc_data = [
            {"title": "Intro", "page": 1, "structure": "1"},
            {"title": "Conclusion", "page": 10, "structure": "2"},
        ]
        result = add_page_offset_to_toc_json(toc_data, offset=5)
        assert result[0]["physical_index"] == 6
        assert result[1]["physical_index"] == 15
        assert "page" not in result[0]

    def test_skips_items_with_no_page_number(self):
        toc_data = [{"title": "Part I", "page": None, "structure": "1"}]
        result = add_page_offset_to_toc_json(toc_data, offset=5)
        # item without an int page is left untouched, not crashed on
        assert "physical_index" not in result[0] or result[0].get("physical_index") is None


# ---------------------------------------------------------------------------
# llm_structured — finish_reason / truncation handling
# ---------------------------------------------------------------------------

class DummyResponseModel(BaseModel):
    value: str


def _make_mock_completion(finish_reason="stop"):
    mock_choice = MagicMock()
    mock_choice.finish_reason = finish_reason
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


class TestLLMStructuredTruncationHandling:
    @patch("pageindex.utils.sync_instructor_client")
    def test_returns_result_on_normal_completion(self, mock_client):
        expected = DummyResponseModel(value="ok")
        mock_client.chat.completions.create_with_completion.return_value = (
            expected,
            _make_mock_completion(finish_reason="stop"),
        )
        result = llm_structured(
            model="test-model", prompt="hello", response_model=DummyResponseModel
        )
        assert result == expected

    @patch("pageindex.utils.sync_instructor_client")
    def test_raises_on_truncated_completion(self, mock_client):
        """
        Regression test: previously llm_structured had no visibility into
        finish_reason at all (create() discards completion metadata), so a
        truncated response would either fail Pydantic validation with a
        confusing error, or in principle validate against partial/incomplete
        data. This should now fail explicitly and legibly.
        """
        mock_client.chat.completions.create_with_completion.return_value = (
            DummyResponseModel(value="partial"),
            _make_mock_completion(finish_reason="length"),
        )
        with pytest.raises(ValueError, match="truncated"):
            llm_structured(
                model="test-model", prompt="hello", response_model=DummyResponseModel
            )

    @patch("pageindex.utils.sync_instructor_client")
    def test_passes_max_tokens_through(self, mock_client):
        mock_client.chat.completions.create_with_completion.return_value = (
            DummyResponseModel(value="ok"),
            _make_mock_completion(finish_reason="stop"),
        )
        llm_structured(
            model="test-model",
            prompt="hello",
            response_model=DummyResponseModel,
            max_tokens=9000,
        )
        _, kwargs = mock_client.chat.completions.create_with_completion.call_args
        assert kwargs["max_tokens"] == 9000

    @patch("pageindex.utils.sync_instructor_client")
    def test_uses_default_max_tokens_when_not_specified(self, mock_client):
        mock_client.chat.completions.create_with_completion.return_value = (
            DummyResponseModel(value="ok"),
            _make_mock_completion(finish_reason="stop"),
        )
        llm_structured(model="test-model", prompt="hello", response_model=DummyResponseModel)
        _, kwargs = mock_client.chat.completions.create_with_completion.call_args
        assert kwargs["max_tokens"] == 4000  # current default — update if you change it

    @patch("pageindex.utils.sync_instructor_client")
    def test_max_retries_is_capped_at_one(self, mock_client):
        """
        Regression test: max_retries was found to actively degrade output
        quality on smaller/local models by feeding failed completions + raw
        Pydantic errors back into context across retries. Confirms it's not
        silently reverted back to a higher value.
        """
        mock_client.chat.completions.create_with_completion.return_value = (
            DummyResponseModel(value="ok"),
            _make_mock_completion(finish_reason="stop"),
        )
        llm_structured(model="test-model", prompt="hello", response_model=DummyResponseModel)
        _, kwargs = mock_client.chat.completions.create_with_completion.call_args
        assert kwargs["max_retries"] == 1

    @patch("pageindex.utils.sync_instructor_client")
    def test_strips_litellm_prefix_from_model_name(self, mock_client):
        mock_client.chat.completions.create_with_completion.return_value = (
            DummyResponseModel(value="ok"),
            _make_mock_completion(finish_reason="stop"),
        )
        llm_structured(
            model="litellm/gpt-4o", prompt="hello", response_model=DummyResponseModel
        )
        args, kwargs = mock_client.chat.completions.create_with_completion.call_args
        assert kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# toc_transformer — end-to-end with mocked Instructor client
# ---------------------------------------------------------------------------

class TestTocTransformer:
    @patch("pageindex.page_index.llm_structured")
    def test_transforms_toc_content_into_structured_list(self, mock_llm_structured):
        mock_result = TOCTransformation(
            table_of_contents=[
                TOCItem(structure="1", title="Introduction", page=1),
                TOCItem(structure="2", title="Conclusion", page=10),
            ]
        )
        mock_llm_structured.return_value = mock_result

        result = toc_transformer("1. Introduction ... 1\n2. Conclusion ... 10", model="test")

        assert len(result) == 2
        assert result[0]["title"] == "Introduction"
        assert result[0]["page"] == 1
        assert result[1]["title"] == "Conclusion"

    @patch("pageindex.page_index.llm_structured")
    def test_calls_llm_structured_with_generous_max_tokens(self, mock_llm_structured):
        """
        toc_transformer intentionally uses a higher max_tokens than the
        llm_structured default, since TOCs can be long. This guards against
        that being silently dropped in a future edit.
        """
        mock_llm_structured.return_value = TOCTransformation(table_of_contents=[])
        toc_transformer("some toc", model="test")
        _, kwargs = mock_llm_structured.call_args
        assert kwargs.get("max_tokens", 0) >= 8000

    @patch("pageindex.page_index.llm_structured")
    def test_empty_toc_returns_empty_list(self, mock_llm_structured):
        mock_llm_structured.return_value = TOCTransformation(table_of_contents=[])
        result = toc_transformer("no sections here", model="test")
        assert result == []

    @patch("pageindex.page_index.llm_structured")
    def test_propagates_truncation_error(self, mock_llm_structured):
        """
        If the underlying llm_structured call raises due to truncation
        (finish_reason == 'length'), toc_transformer should not swallow it —
        this documents the deliberate behavior change from main's old
        continuation-loop approach (see PR discussion).
        """
        mock_llm_structured.side_effect = ValueError("Response was truncated...")
        with pytest.raises(ValueError, match="truncated"):
            toc_transformer("a very long toc", model="test")


# ---------------------------------------------------------------------------
# AliasChoices — confirm reasoning-field synonym handling actually works
# ---------------------------------------------------------------------------

class TestReasoningFieldAliases:
    """
    Some models (observed with local/smaller models during testing) use a
    synonym key ('reason', 'explanation', 'rationale') instead of 'thinking'
    for free-text reasoning fields. AliasChoices was added so these still
    validate instead of failing with "Field required".
    """

    def test_accepts_canonical_thinking_key(self):
        from pageindex.schemas import TOCDetection
        obj = TOCDetection.model_validate({"thinking": "because X", "toc_detected": "yes"})
        assert obj.thinking == "because X"

    @pytest.mark.parametrize("alias_key", ["reason", "explanation", "rationale"])
    def test_accepts_synonym_keys(self, alias_key):
        from pageindex.schemas import TOCDetection
        obj = TOCDetection.model_validate({alias_key: "because X", "toc_detected": "yes"})
        assert obj.thinking == "because X"

    def test_rejects_missing_reasoning_field_entirely(self):
        from pageindex.schemas import TOCDetection
        with pytest.raises(ValidationError):
            TOCDetection.model_validate({"toc_detected": "yes"})