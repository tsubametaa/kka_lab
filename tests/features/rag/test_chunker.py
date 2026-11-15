"""Tests for app.features.rag.chunker (Markdown heading, table, math, and page chunking)."""

import unittest

from app.features.ocr import PageMarkdown
from app.features.rag import chunk_markdown_pages


class RagChunkerTests(unittest.TestCase):
    def test_markdown_chunker_preserves_headings_tables_math_and_pages(self):
        pages = [
            PageMarkdown(
                page_number=1,
                markdown=(
                    '# Bab 2 Evaluasi Model\n\n'
                    '## Confusion Matrix\n\n'
                    '| Prediksi | Positif | Negatif |\n'
                    '| --- | --- | --- |\n'
                    '| Positif | TP | FP |\n'
                    '| Negatif | FN | TN |\n\n'
                    '$$\nF1 = \\frac{2 \\cdot TP}{2 \\cdot TP + FP + FN}\n$$'
                ),
            ),
            PageMarkdown(
                page_number=2,
                markdown='Nilai F1 menggabungkan presisi dan recall secara harmonis pada data timpang.',
            ),
        ]
        chunks = chunk_markdown_pages(pages, max_tokens=220, overlap_tokens=0)
        self.assertGreaterEqual(len(chunks), 1)
        joined = '\n'.join(c.text for c in chunks)
        self.assertIn('[Bab 2 Evaluasi Model > Confusion Matrix]', joined)
        self.assertIn('| Positif | TP | FP |', joined)
        self.assertIn('F1 = \\frac{2 \\cdot TP}', joined)
        self.assertEqual(chunks[0].topic, 'Confusion Matrix')
        all_pages = set()
        for c in chunks:
            if isinstance(c.pages_pdf, list):
                all_pages.update(c.pages_pdf)
            else:
                all_pages.add(c.pages_pdf)
        self.assertEqual(all_pages, {1, 2})


if __name__ == '__main__':
    unittest.main()
