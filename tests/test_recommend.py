import unittest
from src.models.recommend import recommend_modules

class TestRecommend(unittest.TestCase):
    def test_recommend_modules_stub(self):
        result = recommend_modules("North", ["fever"], "basic")
        self.assertIn('modules', result)

if __name__ == '__main__':
    unittest.main()
