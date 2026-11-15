"""Tests for app.features.language.detector."""

import unittest

from app.features.language import (
    detect_language,
    format_user_message,
    is_wrong_language,
)


class LanguageDetectorTests(unittest.TestCase):
    def test_detect_english_queries_including_indonesian_acronyms(self):
        self.assertEqual(
            detect_language('do you know about "PKTO" in prompting engineering?'),
            'en',
        )
        self.assertEqual(
            detect_language('What is Peran, Konteks, Tujuan, dan Output?'),
            'en',
        )
        self.assertEqual(
            detect_language('Can you explain data latih and data uji?'),
            'en',
        )
        self.assertEqual(detect_language("I don't understand this concept"), 'en')
        self.assertEqual(detect_language('Summarize supervised learning'), 'en')
        self.assertEqual(detect_language('Hello!'), 'en')

    def test_detect_indonesian_queries_including_english_tech_terms(self):
        self.assertEqual(
            detect_language('Apa itu "PKTO" dalam prompting engineering?'),
            'id',
        )
        self.assertEqual(
            detect_language('Jelaskan perbedaan supervised learning dan unsupervised learning'),
            'id',
        )
        self.assertEqual(
            detect_language('Kenapa overfitting terjadi pada training data?'),
            'id',
        )
        self.assertEqual(detect_language('Soal no 2 Bagian a Bab I maksudnya apa?'), 'id')
        self.assertEqual(detect_language('Contohnya?'), 'id')

    def test_history_fallback_and_language_switching(self):
        en_history = [
            {'role': 'user', 'content': 'do you know about "PKTO" in prompting engineering?'},
            {'role': 'assistant', 'content': '1. Peran (Role)... Which element defines format?'},
        ]
        id_history = [
            {'role': 'user', 'content': 'Apa itu PKTO dalam prompt engineering?'},
            {'role': 'assistant', 'content': '1. Peran... Bagian mana yang mengatur format?'},
        ]
        self.assertEqual(detect_language('Output', history=en_history), 'en')
        self.assertEqual(detect_language('Output', history=id_history), 'id')
        self.assertEqual(
            detect_language('Sekarang jelaskan contohnya dalam kalimat sederhana', history=en_history),
            'id',
        )
        self.assertEqual(
            detect_language('Can you give me an example in class?', history=id_history),
            'en',
        )

    def test_explicit_language_request_override(self):
        self.assertEqual(detect_language('Tolong jelaskan PKTO dalam bahasa Inggris'), 'en')
        self.assertEqual(detect_language('Please explain PKTO in Indonesian'), 'id')
        self.assertEqual(detect_language('Apa arti kata overfitting dalam bahasa Inggris?'), 'id')
        self.assertEqual(detect_language('What is the meaning of Peran in Indonesian?'), 'en')
        self.assertEqual(
            detect_language('Can you explain what PKTO stands for in Indonesian?'),
            'en',
        )

    def test_format_user_message_adds_transient_reminder_once(self):
        en_msg = format_user_message('What is PKTO?', 'en')
        self.assertIn('(Reply in English.)', en_msg)
        self.assertEqual(format_user_message(en_msg, 'en'), en_msg)

        id_msg = format_user_message('Apa itu PKTO?', 'id')
        self.assertIn('(Jawab dalam bahasa Indonesia.)', id_msg)
        self.assertEqual(format_user_message(id_msg, 'id'), id_msg)

    def test_is_wrong_language_detects_mismatched_responses(self):
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
        self.assertTrue(is_wrong_language(indonesian_leak, 'en'))
        self.assertFalse(is_wrong_language(valid_english_with_pkto, 'en'))
        self.assertTrue(is_wrong_language(valid_english_with_pkto, 'id'))
        self.assertFalse(is_wrong_language(indonesian_leak, 'id'))


if __name__ == '__main__':
    unittest.main()
