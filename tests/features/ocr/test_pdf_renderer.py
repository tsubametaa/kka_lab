"""Tests for app.features.ocr.pdf_renderer (upload validation, DOCX, HTML, and hybrid PDF)."""

import tempfile
import unittest
import zipfile
from pathlib import Path

import pymupdf

from app.features.ocr import (
    DocumentValidationError,
    render_document_pages,
    validate_upload_metadata,
)


def _write_minimal_docx(path: Path, paragraphs: list[tuple[str, str]]):
    p_xml = []
    for style, text in paragraphs:
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ''
        p_xml.append(f'<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>')
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(p_xml)}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr('[Content_Types].xml', '<Types/>')
        zf.writestr('word/document.xml', doc_xml)


class OcrUploadValidationTests(unittest.TestCase):
    def test_validate_upload_metadata_accepts_pdf_docx_html_and_rejects_invalid(self):
        self.assertEqual(validate_upload_metadata('materi.pdf', 100)[1], '.pdf')
        self.assertEqual(validate_upload_metadata('tugas.docx', 100)[1], '.docx')
        self.assertEqual(validate_upload_metadata('artikel.html', 100)[1], '.html')
        self.assertEqual(validate_upload_metadata('page.htm', 100)[1], '.htm')
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('script.exe', 100)
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('kosong.pdf', 0)
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('besar.pdf', 1024, max_bytes=512)

    def test_render_docx_and_html_extract_structured_markdown(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            docx_path = root / 'modul.docx'
            _write_minimal_docx(docx_path, [
                ('Heading1', 'Pengenalan Kecerdasan Artifisial'),
                ('', 'Machine Learning mempelajari pola dari data latih.'),
            ])
            docx_pages = render_document_pages(docx_path, root / 'out_docx')
            self.assertEqual(len(docx_pages), 1)
            self.assertTrue(docx_pages[0].is_structured_text)
            self.assertIn('# Pengenalan Kecerdasan Artifisial', docx_pages[0].native_text)
            self.assertIn('Machine Learning mempelajari pola', docx_pages[0].native_text)

            html_path = root / 'catatan.html'
            html_path.write_text(
                '<html><head><title>Judul</title><style>body{color:red}</style></head>'
                '<body><h1>Bab 1</h1><p>Data latih dan validasi.</p>'
                '<script>alert(1)</script></body></html>',
                encoding='utf-8',
            )
            html_pages = render_document_pages(html_path, root / 'out_html')
            self.assertEqual(len(html_pages), 1)
            self.assertTrue(html_pages[0].is_structured_text)
            self.assertIn('# Bab 1', html_pages[0].native_text)
            self.assertIn('Data latih dan validasi.', html_pages[0].native_text)
            self.assertNotIn('alert(1)', html_pages[0].native_text)

    def test_render_digital_pdf_uses_structured_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pdf_path = root / 'digital.pdf'
            doc = pymupdf.open()
            page = doc.new_page()
            page.insert_text((72, 80), 'Bab 1 Konsep Dasar Kecerdasan Artifisial', fontsize=18)
            page.insert_text(
                (72, 120),
                'Kecerdasan artifisial adalah kemampuan sistem komputer untuk memproses data, '
                'belajar pola secara otomatis dari contoh, dan membantu pengambilan keputusan.',
                fontsize=12,
            )
            doc.save(pdf_path)
            doc.close()

            pages = render_document_pages(pdf_path, root / 'out_pdf')
            self.assertEqual(len(pages), 1)
            self.assertTrue(pages[0].is_structured_text)
            self.assertIn('Konsep Dasar Kecerdasan Artifisial', pages[0].native_text)


if __name__ == '__main__':
    unittest.main()
