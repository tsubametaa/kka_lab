import os
import unittest
from unittest.mock import patch

from app.features.rag.auto_learner import is_auto_learn_enabled
from app.features.rag.context import build_context
from app.features.rag.retriever import DEFAULT_MIN_SIMILARITY, get_min_similarity, retrieve


class WordTokenizer:
    """Minimal word-level tokenizer supporting encode and decode for tests."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(ids)


class RagPipelineTests(unittest.TestCase):
    def test_retrieve_filters_docs_below_min_similarity(self):
        fake_docs = [
            {'_id': 'low', 'text': 'Materi jauh', 'source_title': 'S0', '$similarity': 0.55},
            {'_id': 'ok', 'text': 'Materi relevan', 'source_title': 'S1', '$similarity': 0.78},
            {'_id': 'edge', 'text': 'Tepat batas', 'source_title': 'S2', '$similarity': 0.70},
        ]
        with (
            patch('app.features.rag.retriever.get_astra_collection', return_value=object()),
            patch('app.features.rag.retriever._fetch_query_docs', return_value=fake_docs),
        ):
            docs = retrieve('Apa itu KKA?', retries=0, backoff=0)
        self.assertEqual([d['_id'] for d in docs], ['ok', 'edge'])

    def test_retrieve_respects_env_and_disable_threshold(self):
        fake_docs = [
            {'_id': 'weak', 'text': 'Lemah', 'source_title': 'S0', '$similarity': 0.42},
        ]
        with patch.dict(os.environ, {'RAG_MIN_SIMILARITY': '0.80'}):
            self.assertAlmostEqual(get_min_similarity(), 0.80)
            with (
                patch('app.features.rag.retriever.get_astra_collection', return_value=object()),
                patch('app.features.rag.retriever._fetch_query_docs', return_value=fake_docs),
            ):
                self.assertEqual(retrieve('query', retries=0, backoff=0), [])
                all_docs = retrieve('query', retries=0, backoff=0, min_similarity=0.0)
                self.assertEqual(len(all_docs), 1)
        self.assertAlmostEqual(DEFAULT_MIN_SIMILARITY, 0.70)

    def test_build_context_strips_metadata_headers_and_truncates_at_sentence(self):
        tok = WordTokenizer()
        docs = [
            {
                '_id': 'doc-1',
                'source_title': 'Modul Prompt',
                '$similarity': 0.91,
                'pages_pdf': [3],
                'text': (
                    'Topik: Prompt Engineering | Sumber: Modul KKA\n'
                    'PKTO memandu siswa menulis prompt secara terstruktur. '
                    'Setiap elemen menentukan peran, konteks, tugas, serta bentuk output yang jelas dan akurat. '
                    'Kalimat tambahan panjang sekali yang seharusnya terpotong bersih sebelum batas akhir token terlampaui.'
                ),
            }
        ]
        ctx, sources = build_context(docs, tok, max_tokens=36)
        self.assertNotIn('[S1]', ctx)
        self.assertNotIn('Topik:', ctx)
        self.assertIn('Materi (Modul Prompt):', ctx)
        self.assertTrue(ctx.endswith('.'))
        self.assertNotIn('Kalimat tambahan', ctx)
        self.assertEqual(sources[0]['label'], 'S1')
        self.assertEqual(sources[0]['pages_pdf'], [3])

    def test_auto_learn_defaults_to_disabled_unless_explicitly_enabled(self):
        env_without_flag = {k: v for k, v in os.environ.items() if k != 'RAG_AUTO_LEARN_ENABLED'}
        with patch.dict(os.environ, env_without_flag, clear=True):
            self.assertFalse(is_auto_learn_enabled())
        with patch.dict(os.environ, {'RAG_AUTO_LEARN_ENABLED': 'true'}):
            self.assertTrue(is_auto_learn_enabled())

    def test_build_context_strips_page_markers_while_keeping_template_placeholders(self):
        tok = WordTokenizer()
        docs = [
            {
                '_id': 'doc-page',
                'source_title': 'Bab 1 & 7B',
                '$similarity': 0.88,
                'text': (
                    'Topik: Berpikir Komputasional\n'
                    'Solusi optimal perlu diuji. [Halaman buku 4] Gunakan [HALAMAN/SCREEN] pada template PKTO.'
                ),
            }
        ]
        ctx, _ = build_context(docs, tok, max_tokens=100)
        self.assertNotIn('[Halaman buku 4]', ctx)
        self.assertIn('[HALAMAN/SCREEN]', ctx)
        self.assertIn('Solusi optimal perlu diuji. Gunakan [HALAMAN/SCREEN]', ctx)


if __name__ == '__main__':
    unittest.main()
