import unittest

from pageindex.utils import convert_physical_index_to_int


class TestConvertPhysicalIndexToInt(unittest.TestCase):
    def test_missing_key_does_not_raise(self) -> None:
        data = [
            {"title": "no physical index"},
            {"title": "has physical index", "physical_index": "<physical_index_5>"},
        ]
        out = convert_physical_index_to_int(data)
        self.assertEqual(out[1]["physical_index"], 5)
