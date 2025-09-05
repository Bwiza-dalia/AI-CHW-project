import unittest
from src.models import content

class TestContent(unittest.TestCase):
    def test_summarize_stub(self):
        result = content.summarize("text", "en")
        self.assertIn('summary', result)
    def test_diagram_prompt_stub(self):
        result = content.diagram_prompt("text", "en")
        self.assertIn('prompt', result)
    def test_qa_over_content_stub(self):
        result = content.qa_over_content("Q", "en")
        self.assertIn('answer', result)
    def test_adaptation_suggestions_stub(self):
        result = content.adaptation_suggestions("text", "en")
        self.assertIn('suggestions', result)

if __name__ == '__main__':
    unittest.main()
