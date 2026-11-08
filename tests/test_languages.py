import unittest

from app.features.language import locales
from app.features.language.locales import en, id

class LanguageLocaleTests(unittest.TestCase):
    def test_indonesian_definitions(self):
        self.assertEqual(id.LANGUAGE_CODE, 'id')
        self.assertEqual(id.LANGUAGE_NAME, 'Indonesian')
        self.assertIn('apa', id.STOP_WORDS)
        self.assertIn('jelaskan', id.STOP_WORDS)
        self.assertIn('maksudnya apa', id.VAGUE_REFERENCES)
        self.assertIn('bisa diperjelas', id.CLARIFICATION_MARKERS)
        self.assertIn('saya tidak yakin', id.UNCERTAINTY_MARKERS)

    def test_english_definitions(self):
        self.assertEqual(en.LANGUAGE_CODE, 'en')
        self.assertEqual(en.LANGUAGE_NAME, 'English')
        self.assertIn('what', en.STOP_WORDS)
        self.assertIn('explain', en.STOP_WORDS)
        self.assertIn('what does this mean', en.VAGUE_REFERENCES)
        self.assertIn('could you clarify', en.CLARIFICATION_MARKERS)
        self.assertIn('i am not sure', en.UNCERTAINTY_MARKERS)

    def test_aggregated_definitions(self):
        self.assertEqual(locales.STOP_WORDS, id.STOP_WORDS | en.STOP_WORDS)
        for marker in id.VAGUE_REFERENCES:
            self.assertIn(marker, locales.VAGUE_REFERENCES)
        for marker in en.VAGUE_REFERENCES:
            self.assertIn(marker, locales.VAGUE_REFERENCES)
        for marker in id.CLARIFICATION_MARKERS:
            self.assertIn(marker, locales.CLARIFICATION_MARKERS)
        for marker in en.CLARIFICATION_MARKERS:
            self.assertIn(marker, locales.CLARIFICATION_MARKERS)
        for marker in id.UNCERTAINTY_MARKERS:
            self.assertIn(marker, locales.UNCERTAINTY_MARKERS)
        for marker in en.UNCERTAINTY_MARKERS:
            self.assertIn(marker, locales.UNCERTAINTY_MARKERS)

if __name__ == '__main__':
    unittest.main()
