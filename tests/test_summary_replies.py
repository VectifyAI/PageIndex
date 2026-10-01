"""Summary replies whose JSON does not parse."""
import pytest

import pageindex.utils as utils


@pytest.mark.parametrize("reply, expected", [
    # unescaped quotes inside the text
    ('{\n    "summary": "A new "herd" of models."\n}', 'A new "herd" of models.'),
    # stray backslash from LaTeX, inside a fence
    ('```json\n{\n    "summary": "Scaling $N^*(C)=AC^\\alpha$ holds."\n}\n```',
     'Scaling $N^*(C)=AC^\\alpha$ holds.'),
    # escaped quote next to an unescaped one
    ('{"summary": "He said \\"yes\\" and "no"."}', 'He said "yes" and "no".'),
    # valid JSON, prose and a missing field behave as before
    ('{"summary": "Plain."}', "Plain."),
    ("Just a sentence.", "Just a sentence."),
    ('{"points": []}', '{"points": []}'),
])
def test_parse_summary(reply, expected):
    assert utils.parse_summary(reply) == expected


def test_parse_title_reads_the_field_of_unparseable_json():
    reply = '{"title": "A "big" page", "summary": "Text."}'
    assert utils.parse_title(reply) == 'A "big" page'
    assert utils.parse_summary(reply) == "Text."


def test_parse_title_still_refuses_a_reply_without_the_field():
    assert utils.parse_title("Just a sentence.") == ""
    assert utils.parse_title(None) == ""
