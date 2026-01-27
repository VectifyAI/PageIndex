import unittest

from pageindex.utils import UsageTracker


class TestUsageTracker(unittest.TestCase):
    def test_add_dict_usage(self) -> None:
        tracker = UsageTracker()
        tracker.add({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, model="test-model")

        self.assertEqual(tracker.calls, 1)
        self.assertEqual(tracker.prompt_tokens, 10)
        self.assertEqual(tracker.completion_tokens, 5)
        self.assertEqual(tracker.total_tokens, 15)
        self.assertEqual(tracker.by_model["test-model"]["calls"], 1)

    def test_ignores_missing_usage(self) -> None:
        tracker = UsageTracker()
        tracker.add(None, model="test-model")
        tracker.add({}, model="test-model")
        self.assertEqual(tracker.calls, 0)
