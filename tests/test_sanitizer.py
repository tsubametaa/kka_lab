import unittest

from app.features.chat.sanitizer import StreamingSanitizer, sanitize_response


class SanitizerTests(unittest.TestCase):
    def test_strips_citations_metadata_raw_ids_and_special_tokens(self):
        dirty = (
            '<konteks_sumber>\n'
            'Topik: Klasifikasi | Sumber: Pertemuan 10\n'
            'KNN memakai jarak ke tetangga terdekat [S1]. '
            'Sebagaimana dicatat sebelumnya [M1, S2], data harus dibersihkan dulu '
            '(doc_id: auto_kka_12ab34ef) (skor: 0.412).<|im_end|>'
        )
        clean = sanitize_response(dirty)
        self.assertEqual(
            clean,
            'KNN memakai jarak ke tetangga terdekat. '
            'Sebagaimana dicatat sebelumnya, data harus dibersihkan dulu.',
        )

    def test_preserves_python_indexing_and_fenced_code_blocks(self):
        text = (
            'Gunakan indeks `nilai[0]` dan variabel `S[1]` dalam contoh berikut [S1]:\n'
            '```python\n'
            'nilai = [80, 90]\n'
            'print(nilai[0])\n'
            '```<｜end▁of▁sentence｜>'
        )
        clean = sanitize_response(text)
        self.assertIn('`nilai[0]`', clean)
        self.assertIn('`S[1]`', clean)
        self.assertIn('print(nilai[0])', clean)
        self.assertNotIn('[S1]', clean)
        self.assertNotIn('end▁of▁sentence', clean)

    def test_streaming_sanitizer_handles_split_tags_metadata_and_code(self):
        chunks = [
            'Topik: Klasifikasi',
            ' & Regresi\n',
            'KNN bekerja memakai jarak',
            ' [S',
            '1]. ',
            'Indeks list tetap aman ',
            '`nilai[0]`',
            '.\n```python\narr = [1, 2]\n',
            'print(arr[0])\n```\n',
            'Semoga jelas!<|im_',
            'end|>',
        ]
        stream_san = StreamingSanitizer()
        emitted = [stream_san.feed(c) for c in chunks]
        emitted.append(stream_san.flush())
        result = ''.join(emitted).strip()
        self.assertNotIn('Topik:', result)
        self.assertNotIn('[S1]', result)
        self.assertNotIn('im_end', result)
        self.assertIn('KNN bekerja memakai jarak.', result)
        self.assertIn('`nilai[0]`', result)
        self.assertIn('print(arr[0])', result)
        self.assertTrue(result.endswith('Semoga jelas!'))

    def test_preserves_space_after_inline_backticks(self):
        raw = 'Tanda `=` mengisi kotak, sedangkan `nilai` saat ini dicek lewat `if nilai == 80:` [S1].'
        expected = 'Tanda `=` mengisi kotak, sedangkan `nilai` saat ini dicek lewat `if nilai == 80:`.'
        self.assertEqual(sanitize_response(raw), expected)

        stream_san = StreamingSanitizer()
        out = ''.join(stream_san.feed(raw[i:i + 7]) for i in range(0, len(raw), 7)) + stream_san.flush()
        self.assertEqual(out.strip(), expected)


if __name__ == '__main__':
    unittest.main()
