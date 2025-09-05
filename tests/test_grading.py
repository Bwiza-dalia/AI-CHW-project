import unittest
from src.models.grading import grade_answer

class TestGrading(unittest.TestCase):
    def test_grade_answer_stub(self):
        result = grade_answer("Q", "ref", "user", "en")
        self.assertIn('score_1_to_5', result)
        self.assertIn('similarity', result)
        self.assertIn('lang', result)

if __name__ == '__main__':
    unittest.main()
