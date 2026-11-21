"""Tests for app.features.dataset.validator (pedagogical rules and sample types)."""

import unittest

from app.features.dataset import validate_rows


def _paired_rows(assistant_content):
    base = {
        'group_id': 'a',
        'context': '[S1] materi',
        'messages': [
            {'role': 'user', 'content': 'apa itu algoritma?'},
            {'role': 'assistant', 'content': assistant_content},
        ],
    }
    return [dict(base, id=mode, mode=mode) for mode in ('umum', 'terstruktur')]


class DatasetValidatorTests(unittest.TestCase):
    def test_accepts_valid_paired_lesson_rows(self):
        rows = _paired_rows(
            'Algoritma itu urutan langkah logis untuk menyelesaikan masalah karena komputer butuh instruksi jelas. '
            'Misalnya seperti resep memasak nasi goreng yang berurutan dari menumis bumbu sampai matang. '
            'Menurutmu apa yang terjadi jika langkahnya dibalik?'
        )
        validate_rows(rows)

    def test_rejects_stiff_openers_short_answers_and_missing_pedagogy_elements(self):
        stiff_rows = _paired_rows(
            'Pahami bahwa algoritma itu penting karena komputer butuh aturan. '
            'Misalnya resep masakan yang berurutan. Menurutmu kenapa urutan itu penting?'
        )
        with self.assertRaisesRegex(ValueError, 'pembuka perintah kaku'):
            validate_rows(stiff_rows)

        raw_answer_rows = _paired_rows('Jawab langsung.')
        with self.assertRaisesRegex(ValueError, 'minimal dua kalimat'):
            validate_rows(raw_answer_rows)

        no_reason_rows = _paired_rows(
            'Algoritma merupakan urutan langkah kerja suatu sistem program di komputer sehari-hari. '
            'Misalnya seperti daftar menu makanan di kantin sekolah. Menurutmu apa bedanya?'
        )
        with self.assertRaisesRegex(ValueError, 'menjelaskan alasan'):
            validate_rows(no_reason_rows)

        valid_rows = _paired_rows(
            'Algoritma itu urutan langkah logis untuk menyelesaikan masalah karena komputer butuh instruksi jelas. '
            'Misalnya seperti resep memasak nasi goreng yang berurutan dari menumis bumbu sampai matang. '
            'Menurutmu apa yang terjadi jika langkahnya dibalik?'
        )
        no_example_umum = [
            dict(
                valid_rows[0],
                messages=[
                    valid_rows[0]['messages'][0],
                    {
                        'role': 'assistant',
                        'content': (
                            'Algoritma merupakan urutan langkah terstruktur untuk menyelesaikan suatu tugas komputasi. '
                            'Hal ini penting karena mesin hanya mengeksekusi instruksi secara harfiah. Menurutmu apa manfaatnya?'
                        ),
                    },
                ],
            ),
            valid_rows[1],
        ]
        with self.assertRaisesRegex(ValueError, 'contoh konkret'):
            validate_rows(no_example_umum)

        no_prompt_rows = _paired_rows(
            'Algoritma adalah urutan langkah terstruktur karena komputer butuh instruksi jelas saat bekerja. '
            'Misalnya seperti resep membuat teh manis yang rapi dari awal sampai akhir.'
        )
        with self.assertRaisesRegex(ValueError, 'pemantik berpikir'):
            validate_rows(no_prompt_rows)

        jump_question_rows = _paired_rows(
            'AI bisa membedakan gambar ikan dan ayam karena adanya pelatihan data dan labeling data. '
            'Misalnya komputer mempelajari ratusan contoh foto ayam dan ikan yang sudah diberi nama kategori. '
            'Apakah kamu tahu apa itu labeling dan bagaimana cara kerja nya?'
        )
        with self.assertRaisesRegex(ValueError, 'terlalu loncat/sulit sekaligus'):
            validate_rows(jump_question_rows)

    def test_rejects_duplicate_ids_and_artifact_leaks(self):
        valid_rows = _paired_rows(
            'Algoritma itu urutan langkah logis untuk menyelesaikan masalah karena komputer butuh instruksi jelas. '
            'Misalnya seperti resep memasak nasi goreng yang berurutan dari menumis bumbu sampai matang. '
            'Menurutmu apa yang terjadi jika langkahnya dibalik?'
        )
        with self.assertRaises(ValueError):
            validate_rows(valid_rows + [valid_rows[0]])

        leaked_rows = _paired_rows(
            'Algoritma itu urutan langkah logis karena komputer butuh instruksi jelas [S1]. '
            'Misalnya resep masakan. Menurutmu apa dampaknya jika langkah dibalik?'
        )
        with self.assertRaisesRegex(ValueError, 'artefak teknis'):
            validate_rows(leaked_rows)

    def test_accepts_concise_and_refusal_sample_types(self):
        concise_rows = [
            {
                'id': f'concise-{m}',
                'group_id': 'concise-g1',
                'mode': m,
                'context': 'Materi: PKTO terdiri dari Peran, Konteks, Tugas, dan Output.',
                'sample_type': 'concise',
                'messages': [
                    {'role': 'user', 'content': 'Singkatnya apa kepanjangan PKTO?'},
                    {
                        'role': 'assistant',
                        'content': 'PKTO adalah singkatan dari Peran, Konteks, Tugas, dan Output.',
                    },
                ],
            }
            for m in ('umum', 'terstruktur')
        ]
        validate_rows(concise_rows)

        refusal_rows = [
            {
                'id': f'refusal-{m}',
                'group_id': 'refusal-g1',
                'mode': m,
                'context': '',
                'sample_type': 'refusal',
                'messages': [
                    {'role': 'user', 'content': 'Apa kepanjangan XYZ-99?'},
                    {
                        'role': 'assistant',
                        'content': 'Istilah XYZ-99 tidak ditemukan di materi KKA saat ini; silakan konfirmasi kepada guru.',
                    },
                ],
            }
            for m in ('umum', 'terstruktur')
        ]
        validate_rows(refusal_rows)


if __name__ == '__main__':
    unittest.main()
