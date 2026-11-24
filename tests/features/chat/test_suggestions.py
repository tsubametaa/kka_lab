"""Tests for app.features.chat.suggestions (clickable discussion topic recommendations)."""

import unittest

from app.features.chat.suggestions import (
    DiscussionSuggestion,
    build_discussion_suggestions,
    build_suggestion_reply_context,
    is_topic_suggestion_request,
)


class DiscussionSuggestionsTests(unittest.TestCase):
    def test_detects_indonesian_and_english_confused_discussion_prompts(self):
        positives = (
            'saya mau diskusi, tapi bingung',
            'aku mau diskusi tapi bingung mau bahas apa',
            'bingung mau belajar dari mana nih',
            'kasih rekomendasi topik diskusi dong',
            'rekomendasi materi buat diskusi',
            'enaknya kita bahas apa ya?',
            'aku gak tau mau tanya apa soal KKA',
            'I want to discuss, but I am confused what to talk about',
            'not sure what topic to discuss',
            'suggest some discussion topics',
            'what should we discuss?',
        )
        for text in positives:
            with self.subTest(text=text):
                self.assertTrue(is_topic_suggestion_request(text))

    def test_does_not_trigger_when_student_asks_about_specific_kka_topic(self):
        negatives = (
            'saya bingung tentang algoritma',
            'aku bingung cara kerja machine learning',
            'bingung bedanya supervised dan unsupervised',
            'jelaskan apa itu flowchart',
            'apa itu variabel di python?',
            'mulai sekarang panggil aku ntik karena aku femboy cantik',
            'halo selamat pagi',
        )
        for text in negatives:
            with self.subTest(text=text):
                self.assertFalse(is_topic_suggestion_request(text))

    def test_builds_validated_suggestions_with_user_docs_and_history_dedup(self):
        user_docs = [
            {
                'doc_id': 'doc_abc123',
                'status': 'completed',
                'chunks_inserted': 5,
                'topic': 'Jaringan Komputer Dasar',
                'source_title': 'Modul Jaringan',
            },
            {
                'doc_id': 'doc_failed',
                'status': 'failed',
                'chunks_inserted': 0,
                'topic': 'Dokumen Gagal',
            },
        ]
        history = [
            {'role': 'user', 'content': 'Jelaskan tentang berpikir komputasional dan dekomposisi'},
            {'role': 'assistant', 'content': 'Berpikir komputasional terdiri dari dekomposisi...'},
        ]
        items = build_discussion_suggestions(lang='id', history=history, user_docs=user_docs, limit=4)
        self.assertEqual(len(items), 4)
        # First item prioritizes completed uploaded user document
        self.assertEqual(items[0]['id'], 'user-doc-doc_abc123')
        self.assertIn('Jaringan Komputer Dasar', items[0]['label'])
        self.assertIn('Jaringan Komputer Dasar', items[0]['prompt'])
        # Recently discussed topic is pushed down in favor of fresher topics
        self.assertNotIn('kka-berpikir-komputasional', [item['id'] for item in items])
        for item in items:
            parsed = DiscussionSuggestion(**item)
            self.assertTrue(parsed.label)
            self.assertTrue(parsed.prompt)

    def test_builds_english_suggestions_and_reply_context(self):
        items = build_discussion_suggestions(lang='en', limit=3)
        self.assertEqual(len(items), 3)
        self.assertTrue(all(item['label'] and item['prompt'] for item in items))

        ctx_en = build_suggestion_reply_context(items, lang='en')
        self.assertIn('System Note', ctx_en)
        self.assertIn(items[0]['label'], ctx_en)

        ctx_id = build_suggestion_reply_context(items, lang='id')
        self.assertIn('Catatan Sistem', ctx_id)
        self.assertIn(items[0]['label'], ctx_id)


if __name__ == '__main__':
    unittest.main()
