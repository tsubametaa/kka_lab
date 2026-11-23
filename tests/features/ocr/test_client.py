"""Tests for app.features.ocr.client (vLLM OCR parsing and markdown cleaning)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.features.ocr import (
    DocumentValidationError,
    RenderedPage,
    clean_ocr_markdown,
    parse_pages_to_markdown,
)
from tests.fakes import FakeHttpResponse


class OcrClientTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_vllm_ocr_client_and_markdown_cleaner(self):
        dirty = '```markdown\n# Judul\n\nIsi materi.<｜end▁of▁sentence｜>\n```'
        self.assertEqual(clean_ocr_markdown(dirty), '# Judul\n\nIsi materi.')

        img_path = Path(self.temp_dir.name) / 'page_0001.png'
        img_path.write_bytes(b'\x89PNG\r\n\x1a\nfake')
        rendered = [RenderedPage(page_number=1, image_path=img_path, native_text='')]

        captured_req = {}

        def fake_urlopen(req, timeout=0):
            captured_req['url'] = req.full_url
            captured_req['body'] = json.loads(req.data.decode('utf-8'))
            return FakeHttpResponse({
                'choices': [{'message': {'content': '# Bab 1\n\nTeks hasil OCR.<｜end▁of▁sentence｜>'}}]
            })

        with patch('urllib.request.urlopen', side_effect=fake_urlopen):
            out = parse_pages_to_markdown(rendered, backend='vllm')
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].markdown, '# Bab 1\n\nTeks hasil OCR.')
        self.assertTrue(captured_req['url'].endswith('/chat/completions'))
        self.assertEqual(captured_req['body']['model'], 'baidu/Unlimited-OCR')

        with self.assertRaises(DocumentValidationError):
            parse_pages_to_markdown(rendered, backend='native_pdf')

    def test_structured_text_bypasses_vllm_and_cleaner_strips_det_and_loops(self):
        noisy = (
            '<|det|>image [0, 0, 999, 999]\n'
            'Isi materi asli.\n'
            '[Non-Text] [Non-Text] [Non-Text] [Non-Text]\n'
            '1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12.\n'
            'M. A. K. A. H. A. H. A. H. A.\n'
            'Baris ulang\nBaris ulang\nBaris ulang\nBaris ulang'
        )
        cleaned = clean_ocr_markdown(noisy)
        self.assertNotIn('<|det|>', cleaned)
        self.assertNotIn('[Non-Text]', cleaned)
        self.assertNotIn('11. 12.', cleaned)
        self.assertNotIn('M. A. K.', cleaned)
        self.assertEqual(cleaned.count('Baris ulang'), 2)
        self.assertIn('Isi materi asli.', cleaned)

        structured_page = RenderedPage(
            page_number=1, image_path=None,
            native_text='# Judul DOCX\n\nKonten terstruktur.', is_structured_text=True,
        )
        with patch('urllib.request.urlopen') as mock_open:
            out = parse_pages_to_markdown([structured_page], backend='vllm')
        mock_open.assert_not_called()
        self.assertEqual(out[0].markdown, '# Judul DOCX\n\nKonten terstruktur.')


if __name__ == '__main__':
    unittest.main()
