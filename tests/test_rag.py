import os
import unittest
from unittest.mock import patch

from app.features.rag import RagUnavailableError, retrieve


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


class FakeCollection:
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


class FakeDB:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, *args, **kwargs):
        return self.collection


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_database(self, *args, **kwargs):
        return FakeDB(self.collection)


class RagTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            'ASTRA_DB_API_ENDPOINT': 'https://example-astra-endpoint.apps.astra.datastax.com',
            'ASTRA_DB_APPLICATION_TOKEN': 'AstraCS:real_token_value',
        })
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_retries_transient_503_on_lazy_cursor_and_recovers(self):
        collection = FakeCollection([
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
        collection = FakeCollection([
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
        collection = FakeCollection([
            FakeHttpError(401, '401 Unauthorized'),
        ])
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(collection)), \
             patch('app.features.rag.retriever.load_dotenv'):
            with self.assertRaises(RagUnavailableError) as ctx:
                retrieve('Apa itu data latih?', retries=2, backoff=0)
        self.assertEqual(collection.calls, 1)
        self.assertIn('ASTRA_DB_APPLICATION_TOKEN', str(ctx.exception))

    def test_deduplicates_and_ranks_bilingual_queries(self):
        collection = FakeCollection([
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
        collection = FakeCollection([
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


if __name__ == '__main__':
    unittest.main()

