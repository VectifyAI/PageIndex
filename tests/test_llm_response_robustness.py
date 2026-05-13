"""Tests for LLM response robustness fixes (issues #257 and #199)."""
import json
import re
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from pageindex.utils import extract_json


# --- Issue #257: prompt JSON format has commas after "thinking" ---

def _get_prompt_snippets():
    """Extract reply-format blocks from page_index.py prompts."""
    src = (Path(__file__).parent.parent / "pageindex" / "page_index.py").read_text()
    # Find all JSON-like reply format blocks in prompts
    blocks = re.findall(r'\{\{(.*?)\}\}', src, re.DOTALL)
    return blocks


def test_prompt_thinking_fields_have_trailing_commas():
    """Every 'thinking' field in prompt reply formats must end with a comma."""
    blocks = _get_prompt_snippets()
    assert blocks, "No reply format blocks found in page_index.py"
    for block in blocks:
        lines = block.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('"thinking"') and not stripped.endswith(','):
                # Check it's not the last field (last field before closing brace)
                remaining = [l.strip() for l in lines[i+1:] if l.strip()]
                if remaining:  # there are more fields after this one
                    raise AssertionError(
                        f"'thinking' field missing trailing comma in block:\n{block}"
                    )


# --- Issue #199: generate_toc_init / generate_toc_continue return list ---

def test_generate_toc_init_returns_list_when_llm_returns_dict():
    """generate_toc_init must return [] when extract_json returns a dict, not crash."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"error": "unexpected dict"}'
    mock_response.choices[0].finish_reason = "stop"

    with patch("pageindex.utils.litellm.completion", return_value=mock_response):
        from pageindex.page_index import generate_toc_init
        result = generate_toc_init("some page text", model="gpt-4o")
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"


def test_generate_toc_continue_returns_list_when_llm_returns_dict():
    """generate_toc_continue must return [] when extract_json returns a dict, not crash."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"error": "unexpected dict"}'
    mock_response.choices[0].finish_reason = "stop"

    with patch("pageindex.utils.litellm.completion", return_value=mock_response):
        from pageindex.page_index import generate_toc_continue
        existing_toc = [{"structure": "1", "title": "Intro", "physical_index": "<physical_index_1>"}]
        result = generate_toc_continue(existing_toc, "next page text", model="gpt-4o")
        assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"


def test_generate_toc_init_returns_list_when_llm_returns_valid_list():
    """generate_toc_init passes through a valid list unchanged."""
    toc_list = [{"structure": "1", "title": "Chapter 1", "physical_index": "<physical_index_1>"}]
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(toc_list)
    mock_response.choices[0].finish_reason = "stop"

    with patch("pageindex.utils.litellm.completion", return_value=mock_response):
        from pageindex.page_index import generate_toc_init
        result = generate_toc_init("some page text", model="gpt-4o")
        assert result == toc_list


def test_process_no_toc_does_not_crash_on_dict_response():
    """process_no_toc must not raise AttributeError when generate_toc_init returns []."""
    # When generate_toc_init returns [], the for loop over group_texts[1:] is skipped
    # and toc_with_page_number stays as [], so .extend() is never called on a dict.
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"error": "malformed"}'
    mock_response.choices[0].finish_reason = "stop"

    logger = MagicMock()

    with patch("pageindex.utils.litellm.completion", return_value=mock_response), \
         patch("pageindex.utils.litellm.token_counter", return_value=10):
        from pageindex.page_index import process_no_toc
        page_list = [("Page one content",)]
        result = process_no_toc(page_list, start_index=1, model="gpt-4o", logger=logger)
        assert isinstance(result, list)
