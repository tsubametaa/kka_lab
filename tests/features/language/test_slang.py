"""Tests for bilingual chat slang normalization and cross-feature integration."""

import unittest

from app.features.chat.suggestions import is_topic_suggestion_request
from app.features.language import (
    detect_language,
    format_user_message,
    has_slang,
    normalize_slang,
)
from app.features.language.translator import build_bilingual_query_messages
from app.features.persona.extractor import detect_persona_command
from app.features.rag.context import filter_relevant_history, retrieval_query


class SlangNormalizationTests(unittest.TestCase):
    def test_normalizes_requested_english_and_indonesian_slang_examples(self):
        self.assertEqual(normalize_slang('idk'), "I don't know")
        self.assertEqual(normalize_slang('dunno'), "don't know")
        self.assertEqual(normalize_slang('r u know abt tht'), 'do you know about that')
        self.assertEqual(normalize_slang('tell me smth abt AI'), 'tell me something about AI')
        self.assertEqual(normalize_slang('skrg'), 'sekarang')
        self.assertEqual(normalize_slang('gtw'), 'tidak tahu')
        self.assertEqual(normalize_slang('knp'), 'kenapa')
        self.assertEqual(
            normalize_slang('gtw knp skrg gmn yg dgn AI'),
            'tidak tahu kenapa sekarang bagaimana yang dengan AI',
        )
        self.assertTrue(has_slang('r u know abt tht'))
        self.assertTrue(has_slang('gtw knp'))
        self.assertFalse(has_slang('Apa itu kecerdasan artifisial?'))

    def test_preserves_code_blocks_urls_titles_units_and_kka_acronyms(self):
        code_msg = 'knp kode `u = skrg + 1` error?\n```python\nfor u in range(5):\n    skrg = u + 1\n```'
        normalized = normalize_slang(code_msg)
        self.assertTrue(normalized.startswith('kenapa kode `u = skrg + 1` error?'))
        self.assertIn('for u in range(5):\n    skrg = u + 1', normalized)

        url_msg = 'cek https://example.com/skrg/idk/gtw ya'
        self.assertIn('https://example.com/skrg/idk/gtw', normalize_slang(url_msg))

        self.assertEqual(normalize_slang('Dr. Budi berjalan 10 km ke SMA'), 'Dr. Budi berjalan 10 km ke SMA')
        self.assertEqual(normalize_slang('Jika u = 5 maka u + 2 = 7'), 'Jika u = 5 maka u + 2 = 7')
        self.assertEqual(normalize_slang('Apa itu AI, ML, DL, dan PKTO?'), 'Apa itu AI, ML, DL, dan PKTO?')

    def test_language_detector_handles_english_and_indonesian_slang(self):
        self.assertEqual(detect_language('idk'), 'en')
        self.assertEqual(detect_language('dunno'), 'en')
        self.assertEqual(detect_language('r u know abt tht'), 'en')
        self.assertEqual(detect_language('tell me smth abt machine learning'), 'en')
        self.assertEqual(detect_language('gtw knp'), 'id')
        self.assertEqual(detect_language('skrg gmn caranya yg bener?'), 'id')

    def test_format_user_message_adds_transient_slang_hint_only_when_needed(self):
        slang_en = format_user_message('r u know abt tht?', 'en')
        self.assertIn('do you know about that?', slang_en)
        self.assertIn('(Reply in English.)', slang_en)
        self.assertEqual(format_user_message(slang_en, 'en'), slang_en)

        slang_id = format_user_message('knp overfitting bs terjadi?', 'id')
        self.assertIn('kenapa overfitting bisa terjadi?', slang_id)
        self.assertIn('(Jawab dalam bahasa Indonesia.)', slang_id)
        self.assertEqual(format_user_message(slang_id, 'id'), slang_id)

        plain_id = format_user_message('Apa itu algoritma?', 'id')
        self.assertNotIn('Makna pesan baku', plain_id)

    def test_bilingual_query_and_rag_context_understand_slang_followups(self):
        msgs, condensed = build_bilingual_query_messages('knp overfitting bs terjadi?')
        self.assertEqual(condensed, 'kenapa overfitting bisa terjadi?')
        self.assertIn('kenapa overfitting bisa terjadi?', msgs[-1]['content'])

        history = [
            {'role': 'user', 'content': 'What is data labeling?'},
            {'role': 'assistant', 'content': 'Data labeling tags examples. What happens if labels are wrong?'},
        ]
        # Student replies with English or Indonesian slang "idk" / "dunno" / "gtw"
        for reply in ('idk', 'dunno', 'gtw'):
            with self.subTest(reply=reply):
                kept = filter_relevant_history(reply, history)
                self.assertEqual(len(kept), 2)
                query = retrieval_query(reply, history[0]['content'], history[1]['content'])
                self.assertIn('data labeling', query.lower())

    def test_persona_and_suggestions_understand_slang_inputs(self):
        cmd = detect_persona_command('mulai skrg panggil gw ntik krn gw femboy cantik')
        self.assertEqual(cmd.action, 'set')
        self.assertIn('ntik', cmd.persona)

        self.assertTrue(is_topic_suggestion_request('gw mau diskusi tp gtw mau bahas apa'))
        self.assertTrue(is_topic_suggestion_request('idk what to discuss'))
        self.assertTrue(is_topic_suggestion_request('dunno what to talk abt'))
        # Still avoids false positive when a specific KKA term is mentioned with slang
        self.assertFalse(is_topic_suggestion_request('gw gtw knp overfitting terjadi'))


if __name__ == '__main__':
    unittest.main()
