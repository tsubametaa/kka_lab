"""Tests for app.features.rag.retriever (retry logic, ranking, similarity floor, scope filtering)."""

import os
import unittest
from unittest.mock import patch

from app.features.rag import RagUnavailableError, build_scope_filter, retrieve
from app.features.rag.retriever import DEFAULT_MIN_SIMILARITY, get_min_similarity
from tests.fakes import FakeClient, RecordingAstraCollection


class FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class FakeHttpError(Exception):
    def __init__(self, status_code, message='HTTP error'):
        super().__init__(message)
        self.response = FakeResponse(status_code)


class FailingCursor:
    """Meniru astrapy FindCursor yang baru melempar error saat diiterasi."""

    def __init__(self, exc):
        self.exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        raise self.exc


class SequencedCollection:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0
        self.queries_seen = []

    def find(self, *args, **kwargs):
        self.calls += 1
        sort_arg = kwargs.get('sort') or {}
        if '$vectorize' in sort_arg:
            self.queries_seen.append(sort_arg['$vectorize'])
        effect = self.side_effects.pop(0)
        if isinstance(effect, Exception):
            return FailingCursor(effect)
        return iter(effect)


class RagRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            'ASTRA_DB_API_ENDPOINT': 'https://example-astra-endpoint.apps.astra.datastax.com',
            'ASTRA_DB_APPLICATION_TOKEN': 'AstraCS:real_token_value',
        })
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_retries_transient_503_on_lazy_cursor_and_recovers(self):
        collection = SequencedCollection([
            FakeHttpError(503, '503 Service Unavailable'),
            [{'_id': '1', 'text': 'Materi data latih', 'source_title': 'Bab 1', '$similarity': 0.91}],
        ])
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(collection)), \
             patch('app.features.rag.retriever.load_dotenv'):
            docs = retrieve('Apa itu data latih?', retries=2, backoff=0)
        self.assertEqual(collection.calls, 2)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['_id'], '1')

    def test_persistent_503_raises_rag_unavailable_with_hibernation_hint(self):
        collection = SequencedCollection([
            FakeHttpError(503, '503 Service Unavailable'),
            FakeHttpError(503, '503 Service Unavailable'),
            FakeHttpError(503, '503 Service Unavailable'),
        ])
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(collection)), \
             patch('app.features.rag.retriever.load_dotenv'):
            with self.assertRaises(RagUnavailableError) as ctx:
                retrieve('Apa itu data latih?', retries=2, backoff=0)
        self.assertEqual(collection.calls, 3)
        self.assertIn('503 Service Unavailable', str(ctx.exception))
        self.assertIn('Hibernated', str(ctx.exception))

    def test_auth_401_does_not_retry(self):
        collection = SequencedCollection([
            FakeHttpError(401, '401 Unauthorized'),
        ])
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(collection)), \
             patch('app.features.rag.retriever.load_dotenv'):
            with self.assertRaises(RagUnavailableError) as ctx:
                retrieve('Apa itu data latih?', retries=2, backoff=0)
        self.assertEqual(collection.calls, 1)
        self.assertIn('ASTRA_DB_APPLICATION_TOKEN', str(ctx.exception))

    def test_deduplicates_and_ranks_bilingual_queries(self):
        collection = SequencedCollection([
            [
                {'_id': 'a', 'text': 'Teks A', 'source_title': 'S1', '$similarity': 0.75},
                {'_id': 'b', 'text': 'Teks B', 'source_title': 'S2', '$similarity': 0.82},
            ],
            [
                {'_id': 'a', 'text': 'Teks A', 'source_title': 'S1', '$similarity': 0.93},
            ],
        ])
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(collection)), \
             patch('app.features.rag.retriever.load_dotenv'):
            docs = retrieve(['Apa itu data latih?', 'What is training data?'], top_k=2, retries=0, backoff=0)
        self.assertEqual([d['_id'] for d in docs], ['a', 'b'])
        self.assertEqual(docs[0]['$similarity'], 0.93)

    def test_retrieve_condenses_very_long_query_keeping_head_and_tail(self):
        collection = SequencedCollection([
            [{'_id': '1', 'text': 'Materi PKTO', 'source_title': 'Bab 5', '$similarity': 0.88}],
        ])
        long_query = 'TOPIK_AWAL_SURAT ' + ('x' * 1500) + ' PERTANYAAN_AKHIR_PKTO'
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(collection)), \
             patch('app.features.rag.retriever.load_dotenv'):
            retrieve(long_query, retries=0, backoff=0)
        self.assertEqual(len(collection.queries_seen), 1)
        sent = collection.queries_seen[0]
        self.assertLessEqual(len(sent), 1001)
        self.assertIn('TOPIK_AWAL_SURAT', sent)
        self.assertIn('PERTANYAAN_AKHIR_PKTO', sent)

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

    def test_scope_filter_isolates_private_student_documents(self):
        anon_filter = build_scope_filter(None)
        self.assertEqual(anon_filter, {
            '$or': [
                {'metadata.scope': {'$exists': False}},
                {'metadata.scope': 'global'},
            ]
        })

        student_filter = build_scope_filter('siswa-1')
        self.assertIn(
            {'metadata.scope': 'user', 'metadata.uploaded_by': 'siswa-1'},
            student_filter['$or'],
        )
        self.assertNotIn(
            {'metadata.scope': 'user', 'metadata.uploaded_by': 'siswa-2'},
            student_filter['$or'],
        )

        col = RecordingAstraCollection()
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            retrieve('Apa itu overfitting?', user_id='siswa-1', retries=0, backoff=0)
        self.assertEqual(col.find_filters[0], student_filter)


if __name__ == '__main__':
    unittest.main()
