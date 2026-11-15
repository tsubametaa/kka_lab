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


if __name__ == '__main__':
    unittest.main()
