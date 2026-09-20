import json
from pathlib import Path

import pytest

from pageindex.naming import (
    sanitize_filename,
    truncate_filename,
    validate_folder_name,
    validate_stored_filename,
)

CASES = json.loads((Path(__file__).parent / "fixtures/naming-v1.json").read_text())


@pytest.mark.parametrize("case", CASES["files"])
def test_upload_contract(case):
    result = sanitize_filename(case["name"])
    assert result == case["expected"]
    assert len(result.encode("utf-8")) <= 180
    assert sanitize_filename(result) == result
    assert validate_folder_name(result) == result


@pytest.mark.parametrize("case", CASES["folders"])
def test_folder_contract(case):
    if "error" in case:
        with pytest.raises(ValueError, match=case["error"]):
            validate_folder_name(case["name"])
    else:
        assert validate_folder_name(case["name"]) == case["expected"]


@pytest.mark.parametrize("case", CASES["suffixes"])
def test_collision_contract(case):
    assert truncate_filename(case["name"], suffix=case["suffix"]) == case["expected"]


@pytest.mark.parametrize(
    "name", ["中" * 200 + ".pdf", "😀" * 100 + ".pdf", "a." + "b" * 300]
)
def test_byte_budget_includes_extension_and_collision_suffix(name):
    stored = sanitize_filename(name)
    candidates = [stored] + [
        truncate_filename(stored, suffix=f"_{i}") for i in range(1, 100)
    ]
    assert len(set(candidates)) == 100
    for candidate in candidates:
        assert len(candidate.encode("utf-8")) <= 180
        assert validate_folder_name(candidate) == candidate
    if name.endswith(".pdf"):
        assert all(candidate.endswith(".pdf") for candidate in candidates)


@pytest.mark.parametrize(
    "name", ["x" * 200 + ".pdf", " Report  2026.pdf", "Q3:2026?.pdf"]
)
def test_assigned_names_are_read_literally(name):
    assert validate_stored_filename(name) == name


@pytest.mark.parametrize(
    "name", ["../report.pdf", "a/b.pdf", "a\\b.pdf", "..", "a\n.pdf"]
)
def test_stored_name_cannot_escape_its_directory(name):
    with pytest.raises(ValueError):
        validate_stored_filename(name)
