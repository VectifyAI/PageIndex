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


def test_latex_in_invalid_json_keeps_its_backslashes():
    reply = r'{"summary": "Shows $\alpha \to \nu$ as $L\times W$ \u2190 width.\n\nThen \"more\"."}'
    assert utils.parse_summary(reply) == 'Shows $\\alpha \\to \\nu$ as $L\\times W$ ← width.\n\nThen "more".'


def test_latex_that_json_decodes_to_control_characters_is_read_raw():
    reply = r'{"summary": "A $W \times L$ grid of $\boldsymbol{x}$ and $\nu$.", "title": "The $\times$ map"}'
    assert utils.parse_summary(reply) == r"A $W \times L$ grid of $\boldsymbol{x}$ and $\nu$."
    assert utils.parse_title(reply) == r"The $\times$ map"


def test_valid_json_with_escaped_latex_and_paragraphs_is_unchanged():
    assert utils.parse_summary(r'{"summary": "First $\\alpha$.\n\nSecond."}') == "First $\\alpha$.\n\nSecond."
    assert utils.parse_summary(r'{"summary": "Costs $5.\n\nLater $10."}') == "Costs $5.\n\nLater $10."


def test_latex_commands_outside_math_are_kept():
    reply = r'{"summary": "A leap (\text{Eq. } \ref{eq:71}).\n\nThen\tDone."}'
    assert utils.parse_summary(reply) == 'A leap (\\text{Eq. } \\ref{eq:71}).\n\nThen\tDone.'
