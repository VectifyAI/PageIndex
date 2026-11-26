from pageindex.utils import extract_json


def test_bare_none_becomes_null():
    assert extract_json('{"physical_index": None}') == {"physical_index": None}


def test_none_inside_a_string_is_kept():
    reply = '{"title": "None of the Above", "physical_index": None}'
    assert extract_json(reply) == {"title": "None of the Above", "physical_index": None}


def test_none_in_fenced_reply_and_escaped_quote():
    reply = '```json\n{"title": "Say \\"None\\" twice", "items": [None, "NoneType"]}\n```'
    assert extract_json(reply) == {"title": 'Say "None" twice', "items": [None, "NoneType"]}
