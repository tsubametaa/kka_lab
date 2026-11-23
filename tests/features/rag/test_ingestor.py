"""Tests for app.features.rag.ingestor (Astra DB vectorize payload and retry ingestion)."""

import os
import unittest
from unittest.mock import patch

from app.features.ocr import PageMarkdown
from app.features.rag import build_astra_documents, chunk_markdown_pages, ingest_chunks
from tests.fakes import FakeClient, RecordingAstraCollection


class RagIngestorTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {
            'ASTRA_DB_API_ENDPOINT': 'https://example-astra.apps.astra.datastax.com',
            'ASTRA_DB_APPLICATION_TOKEN': 'AstraCS:test_token',
        }, clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_ingest_sets_both_text_and_vectorize_with_retry(self):
        pages = [PageMarkdown(page_number=3, markdown='# Regresi\n\nRegresi memprediksi nilai kontinu.')]
        chunks = chunk_markdown_pages(pages, max_tokens=200)
        docs = build_astra_documents(
            chunks,
            doc_id='doc_abc',
            source_title='Catatan Siswa',
            user_id='siswa-1',
            scope='user',
            filename='catatan.pdf',
        )
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]['text'], docs[0]['$vectorize'])
        self.assertEqual(docs[0]['metadata']['scope'], 'user')
        self.assertEqual(docs[0]['metadata']['uploaded_by'], 'siswa-1')
        self.assertEqual(docs[0]['pages_pdf'], 3)

        col = RecordingAstraCollection(fail_first_insert=True)
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            count = ingest_chunks(
                chunks,
                doc_id='doc_abc',
                source_title='Catatan Siswa',
                user_id='siswa-1',
                scope='user',
                retries=2,
                backoff=0,
            )
        self.assertEqual(count, 1)
        self.assertEqual(len(col.inserted_batches), 1)

    def test_ingest_splits_long_chunk_and_retries_on_512_token_error(self):
        from app.features.rag import DocumentChunk

        huge_chunk = DocumentChunk(
            index=0,
            text=' '.join([f'kata{i}' for i in range(500)]),
            topic='Materi',
            pages_pdf=1,
        )
        docs = build_astra_documents(
            [huge_chunk],
            doc_id='doc_big',
            source_title='Besar',
            user_id='siswa-1',
            scope='user',
        )
        self.assertGreater(len(docs), 1)
        for d in docs:
            self.assertLessEqual(len(d['$vectorize']), 1100)

        class RejectLongOnceCollection(RecordingAstraCollection):
            def __init__(self):
                super().__init__()
                self.rejected = False

            def insert_many(self, batch, **kwargs):
                if not self.rejected and any(len(d['$vectorize']) > 200 for d in batch):
                    self.rejected = True
                    raise RuntimeError('Input length 600 exceeds maximum allowed token size 512')
                return super().insert_many(batch, **kwargs)

        col = RejectLongOnceCollection()
        medium_chunk = DocumentChunk(
            index=0,
            text=' '.join(['materi'] * 80),
            topic='Materi',
            pages_pdf=1,
        )
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            count = ingest_chunks(
                [medium_chunk],
                doc_id='doc_split',
                source_title='Split',
                user_id='siswa-1',
                scope='user',
                retries=0,
                backoff=0,
            )
        self.assertGreaterEqual(count, 2)


if __name__ == '__main__':
    unittest.main()
