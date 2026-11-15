"""Tests for app.features.language.translator."""

import unittest
from unittest.mock import patch

from app.features.language import (
    bilingual_queries,
    ensure_response_language,
    parse_bilingual_queries,
)


class LanguageTranslatorTests(unittest.TestCase):
    def test_parse_bilingual_queries(self):
        output = '{"id":"Apa itu data latih?","en":"What is training data?"}'
        self.assertEqual(
            parse_bilingual_queries(output, 'What is training data?'),
            ['Apa itu data latih?', 'What is training data?'],
        )

    def test_parse_code_fenced_json_and_remove_duplicates(self):
        output = '```json\n{"id":"Apa itu AI?","en":"Apa itu AI?"}\n```'
        self.assertEqual(parse_bilingual_queries(output, 'Apa itu AI?'), ['Apa itu AI?'])

    def test_invalid_translation_falls_back_to_original(self):
        self.assertEqual(parse_bilingual_queries('not json', 'Pertanyaan asli'), ['Pertanyaan asli'])
        self.assertEqual(parse_bilingual_queries('[]', 'Pertanyaan asli'), ['Pertanyaan asli'])

    def test_ensure_response_language_rewrites_only_when_needed(self):
        indonesian_leak = (
            'Ya, PKTO adalah singkatan dari Peran, Konteks, Tujuan, dan Output. '
            'Ini adalah struktur yang digunakan untuk membuat prompt yang baik '
            'dalam konteks prompting engineering.'
        )
        valid_english_with_pkto = (
            '1. PKTO stands for Peran (Role), Konteks (Context), Tujuan (Objective), '
            'and Output [S1]. 2. Each element clarifies the prompt because clear context '
            'and goals reduce ambiguity. Which PKTO element would you write first, and why?'
        )

        with patch('app.features.language.translator._generate_local_text') as mock_gen:
            result = ensure_response_language(valid_english_with_pkto, 'en', None, None)
            self.assertEqual(result, valid_english_with_pkto)
            mock_gen.assert_not_called()

        with patch(
            'app.features.language.translator._generate_local_text',
            return_value=valid_english_with_pkto,
        ) as mock_gen:
            result = ensure_response_language(indonesian_leak, 'en', object(), object())
            self.assertEqual(result, valid_english_with_pkto)
            mock_gen.assert_called_once()

    def test_bilingual_queries_condenses_very_long_prompt(self):
        long_prompt = 'PEMBUKA_TUGAS ' + ('isi paragraf panjang ' * 60) + ' PERTANYAAN_INTI_SISWA?'
        with patch(
            'app.features.language.translator._generate_local_text',
            return_value='{"id":"tugas pertanyaan inti","en":"task core question"}',
        ) as mock_gen:
            queries = bilingual_queries(long_prompt, object(), object())
        self.assertEqual(queries, ['tugas pertanyaan inti', 'task core question'])
        sent_messages = mock_gen.call_args[0][0]
        sent_user_query = sent_messages[-1]['content']
        self.assertLessEqual(len(sent_user_query), 601)
        self.assertIn('PEMBUKA_TUGAS', sent_user_query)
        self.assertIn('PERTANYAAN_INTI_SISWA?', sent_user_query)


if __name__ == '__main__':
    unittest.main()
