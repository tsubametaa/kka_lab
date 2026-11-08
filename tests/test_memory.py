import tempfile
import unittest

from app.features.memory import ChatStore, normalize_question, training_quality

class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ChatStore(self.temp_dir.name + '/memory.sqlite3')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(
            normalize_question('Apa itu data latih?'),
            normalize_question('APA ITU DATA LATIH!'),
        )

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

    def test_training_gate_needs_consent_and_clear_answer(self):
        question = 'Apa perbedaan data latih dan data uji?'
        answer = 'Data latih digunakan untuk mengajari model, sedangkan data uji '
        answer += 'digunakan untuk mengevaluasi kinerjanya pada contoh baru.'
        self.assertEqual(training_quality(question, answer, True),
                         (True, 'pending_human_review'))
        self.assertEqual(training_quality(question, answer, False),
                         (False, 'consent_not_given'))
        self.assertEqual(training_quality('ini apa?', answer, True),
                         (False, 'question_too_vague'))
        self.assertEqual(
            training_quality('Saya bingung, maksudnya apa ya?', answer, True),
            (False, 'question_unclear'),
        )
        self.assertEqual(
            training_quality('Apakah yang itu benar sekali?', answer, True),
            (False, 'question_unclear'),
        )
        clarification = 'Pertanyaan belum jelas, bisa diperjelas dengan memberi '
        clarification += 'contoh atau konteks yang sedang dipelajari?'
        self.assertEqual(training_quality(question, clarification, True),
                         (False, 'assistant_requested_clarification'))
        uncertain = 'Saya tidak yakin jawaban ini benar; mohon konfirmasi lagi '
        uncertain += 'kepada guru sebelum digunakan.'
        self.assertEqual(training_quality(question, uncertain, True),
                         (False, 'assistant_uncertain'))

        english_question = 'What is the difference between training data and test data?'
        english_answer = (
            'Training data is used to fit a model, while test data evaluates '
            'how it performs on examples it has not seen.'
        )
        self.assertEqual(
            training_quality(english_question, english_answer, True),
            (True, 'pending_human_review'),
        )
        self.assertEqual(
            training_quality('Could you please explain this to me?', english_answer, True),
            (False, 'question_unclear'),
        )
        english_uncertainty = (
            'I am not sure this answer is correct. Please ask your teacher '
            'to confirm the information before using it.'
        )
        self.assertEqual(
            training_quality(english_question, english_uncertainty, True),
            (False, 'assistant_uncertain'),
        )
        self.assertEqual(
            training_quality(
                english_question,
                "I'm not sure this answer is correct, so please ask your teacher.",
                True,
            ),
            (False, 'assistant_uncertain'),
        )

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
