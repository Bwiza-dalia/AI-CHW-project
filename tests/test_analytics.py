import unittest
from src.models.analytics import compute_analytics

class TestAnalytics(unittest.TestCase):
    def test_compute_analytics_stub(self):
        result = compute_analytics()
        self.assertIn('engagement', result)

if __name__ == '__main__':
    unittest.main()
