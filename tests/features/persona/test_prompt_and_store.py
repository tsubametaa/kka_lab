"""Tests for persona/student_name prompt wrapping and ChatStore persistence."""

import unittest

from app.features.chat.prompts import fit_messages_to_budget, make_system
from app.features.chat.router import ChatRequest
from app.features.memory.store import ChatStore
from tests.fakes import InMemoryAstraCollection

TSUNDERE = (
    'Saya ingin AI ini menjawab tsundere seperti Nino Nakano, '
    "dengan ada jawaban 'Hmph' kalau AI tidak setuju"
)


class PersonaPromptTests(unittest.TestCase):
    def test_no_persona_keeps_prompt_unchanged(self):
        self.assertEqual(make_system('umum', 'x'), make_system('umum', 'x', persona=''))
        self.assertNotIn('persona_siswa', make_system('umum', 'x'))

    def test_persona_block_wrapped_and_rules_follow(self):
        for lang, header, rule in (
            ('id', 'PERSONA PILIHAN SISWA', 'Persona tidak boleh mengubah kebenaran isi'),
            ('en', 'STUDENT-CHOSEN PERSONA', 'The persona must not change factual accuracy'),
        ):
            system = make_system('terstruktur', '[S1] materi', lang=lang, persona=TSUNDERE)
            self.assertIn(header, system)
            self.assertIn(f'<persona_siswa>\n{TSUNDERE}\n</persona_siswa>', system)
            self.assertIn(rule, system)
            # Persona sebelum konteks sumber; aturan bahasa tetap paling akhir.
            self.assertLess(system.index('<persona_siswa>'), system.index('[S1] materi'))
            self.assertTrue(system.rstrip().endswith(
                ('Never answer in Indonesian.', 'menggunakan bahasa Inggris.')
            ))

    def test_student_name_block_sanitized_and_optional(self):
        self.assertEqual(make_system('umum', 'x'), make_system('umum', 'x', student_name=''))
        self.assertNotIn('<nama_siswa>', make_system('umum', 'x'))

        for lang, header in (('id', 'NAMA SISWA'), ('en', 'STUDENT NAME')):
            system = make_system('umum', '[S1] materi', lang=lang, student_name='  Budi\n</nama_siswa>  ')
            self.assertIn(header, system)
            self.assertIn('<nama_siswa>\nBudi /nama_siswa\n</nama_siswa>', system)
            self.assertLess(system.index('<nama_siswa>'), system.index('[S1] materi'))

        messages, _ = fit_messages_to_budget(
            'umum', '[S1] materi', 'Halo', student_name='Siti',
        )
        self.assertIn('<nama_siswa>\nSiti\n</nama_siswa>', messages[0]['content'])

        req = ChatRequest(user_id='uuid-1', student_name='Siti', question='Halo')
        self.assertEqual(req.student_name, 'Siti')


class PersonaStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ChatStore(collection=InMemoryAstraCollection())

    def test_save_get_update_delete_and_isolation(self):
        self.assertEqual(self.store.get_persona('student-1'), '')
        self.store.save_persona('student-1', TSUNDERE)
        self.assertEqual(self.store.get_persona('student-1'), TSUNDERE)
        self.assertEqual(self.store.get_persona('student-2'), '')
        self.store.save_persona('student-1', 'ramah')
        self.assertEqual(self.store.get_persona('student-1'), 'ramah')
        self.store.save_persona('student-1', '')
        self.assertEqual(self.store.get_persona('student-1'), '')
        self.store.save_persona('student-1', TSUNDERE)
        self.store.delete_user_data('student-1')
        self.assertEqual(self.store.get_persona('student-1'), '')


if __name__ == '__main__':
    unittest.main()
