import pytest
from pageindex.core.llm import extract_json, count_tokens

def test_extract_json_basic():
    text = '{"key": "value"}'
    assert extract_json(text) == {"key": "value"}

def test_extract_json_with_markdown():
    text = 'Here is the json:\n```json\n{"key": "value"}\n```'
    assert extract_json(text) == {"key": "value"}

def test_extract_json_with_trailing_commas():
    # This might fail depending on implementation robustness, but let's see
    text = '{"key": "value",}'
    # Our implementation tries to fix this
    assert extract_json(text) == {"key": "value"}

def test_count_tokens():
    text = "Hello world"
    # Basic check, exact number depends on encoding
    assert count_tokens(text) > 0
