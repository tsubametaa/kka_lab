"""Tests for app.features.memory.gate (question normalization and training quality gate)."""

import unittest

from app.features.memory import normalize_question, training_quality


class TrainingGateTests(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(
            normalize_question('Apa itu data latih?'),
            normalize_question('APA ITU DATA LATIH!'),
        )

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


if __name__ == '__main__':
    unittest.main()
