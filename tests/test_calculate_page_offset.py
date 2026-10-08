import unittest

from pageindex.page_index_classic import calculate_page_offset


class CalculatePageOffsetTest(unittest.TestCase):
    """Tests for calculate_page_offset function"""

    def test_calculate_offset_with_valid_pairs(self):
        """Should calculate the most common offset from valid page pairs"""
        pairs = [
            {'title': 'Section 1', 'page': 5, 'physical_index': 10},  # offset = 5
            {'title': 'Section 2', 'page': 10, 'physical_index': 15}, # offset = 5
            {'title': 'Section 3', 'page': 15, 'physical_index': 20}, # offset = 5
        ]

        result = calculate_page_offset(pairs)

        self.assertEqual(result, 5)

    def test_calculate_offset_returns_most_common(self):
        """Should return the most common offset when there are variations"""
        pairs = [
            {'title': 'Section 1', 'page': 5, 'physical_index': 10},  # offset = 5
            {'title': 'Section 2', 'page': 10, 'physical_index': 15}, # offset = 5
            {'title': 'Section 3', 'page': 15, 'physical_index': 20}, # offset = 5
            {'title': 'Section 4', 'page': 20, 'physical_index': 24}, # offset = 4 (outlier)
        ]

        result = calculate_page_offset(pairs)

        self.assertEqual(result, 5)  # Most common offset wins

    def test_calculate_offset_with_empty_pairs(self):
        """Should return 0 when given empty pairs list"""
        pairs = []

        result = calculate_page_offset(pairs)

        self.assertEqual(result, 0)

    def test_calculate_offset_with_invalid_pairs(self):
        """Should return 0 when all pairs have missing or invalid data"""
        pairs = [
            {'title': 'Section 1'},  # Missing page and physical_index
            {'title': 'Section 2', 'page': None, 'physical_index': 10},
            {'title': 'Section 3', 'page': 5, 'physical_index': None},
            {'title': 'Section 4', 'page': 'invalid', 'physical_index': 10},
        ]

        result = calculate_page_offset(pairs)

        self.assertEqual(result, 0)

    def test_calculate_offset_with_mixed_valid_invalid(self):
        """Should calculate offset from valid pairs, ignoring invalid ones"""
        pairs = [
            {'title': 'Section 1', 'page': 5, 'physical_index': 10},   # offset = 5
            {'title': 'Section 2'},  # Invalid - missing data
            {'title': 'Section 3', 'page': 10, 'physical_index': 15},  # offset = 5
            {'title': 'Section 4', 'page': None, 'physical_index': 20}, # Invalid
        ]

        result = calculate_page_offset(pairs)

        self.assertEqual(result, 5)

    def test_calculate_offset_always_returns_int(self):
        """Should always return an integer, never None"""
        test_cases = [
            [],  # Empty list
            [{'title': 'Section'}],  # Missing data
            [{'page': None, 'physical_index': None}],  # Null values
        ]

        for pairs in test_cases:
            with self.subTest(pairs=pairs):
                result = calculate_page_offset(pairs)

                self.assertIsNotNone(result, "calculate_page_offset should never return None")
                self.assertIsInstance(result, int, "calculate_page_offset should always return int")
                self.assertEqual(result, 0, "Should return 0 when offset cannot be calculated")


if __name__ == "__main__":
    unittest.main()
