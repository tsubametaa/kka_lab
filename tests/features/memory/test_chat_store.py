"""Tests for app.features.memory.store (session isolation, repeated questions, personal knowledge)."""

import unittest

from app.features.memory import ChatStore
from tests.fakes import InMemoryAstraCollection


class ChatStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ChatStore(collection=InMemoryAstraCollection())

    def test_repeated_question_is_private_to_user(self):
        session = self.store.create_session('student-1')
        self.store.save_exchange(
            'student-1', session, 'Apa perbedaan data latih dan data uji?',
            'Data latih digunakan untuk belajar, sedangkan data uji mengukur hasil.',
            'umum', [], False,
        )
        self.assertIsNotNone(self.store.find_repeated_question(
            'student-1', 'APA PERBEDAAN DATA LATIH DAN DATA UJI!', mode='umum'
        ))
        self.assertIsNone(self.store.find_repeated_question(
            'student-1', 'Apa perbedaan data latih dan data uji?',
            mode='terstruktur',
        ))
        self.assertIsNone(self.store.find_repeated_question(
            'student-2', 'Apa perbedaan data latih dan data uji?'
        ))
        self.store.save_exchange(
            'student-1', session, 'Contohnya?', 'Contoh jawaban panjang tidak relevan.',
            'umum', [], False,
        )
        self.assertIsNone(self.store.find_repeated_question(
            'student-1', 'Contohnya?', mode='umum'
        ))

    def test_personal_knowledge_retrieval_and_isolation(self):
        self.store.save_knowledge(
            'student-1', 'Apa perbedaan data latih dan data uji?',
            'Data latih dipakai untuk menyesuaikan model; data uji untuk evaluasi.',
        )
        matches = self.store.find_relevant_knowledge(
            'student-1', 'Jelaskan data uji dan data latih'
        )
        self.assertEqual(len(matches), 1)
        self.assertIn('data latih', matches[0]['answer'].casefold())
        self.assertEqual(
            self.store.find_relevant_knowledge(
                'student-2', 'Jelaskan data uji dan data latih'
            ),
            [],
        )
        self.assertEqual(self.store.find_relevant_knowledge('student-1', 'Apa itu?'), [])

    def test_session_history_and_delete(self):
        session = self.store.create_session('student-1')
        self.assertTrue(self.store.session_belongs_to(session, 'student-1'))
        self.assertFalse(self.store.session_belongs_to(session, 'student-2'))
        self.store.save_exchange(
            'student-1', session, 'Bagaimana cara belajar materi ini?', 'Jawaban',
            'umum', [], False,
        )
        self.assertEqual(len(self.store.history(session)), 2)
        self.store.delete_user_data('student-1')
        self.assertFalse(self.store.session_belongs_to(session, 'student-1'))
        self.assertIsNone(self.store.find_repeated_question(
            'student-1', 'Bagaimana cara belajar materi ini?'
        ))
        self.store.save_knowledge('student-1', 'Apa itu data baru?', 'Data baru.')
        self.store.delete_user_data('student-1')
        self.assertEqual(
            self.store.find_relevant_knowledge('student-1', 'Jelaskan data baru'), []
        )


if __name__ == '__main__':
    unittest.main()
