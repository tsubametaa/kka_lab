"""Unit tests for Unlimited-OCR parsing, Markdown chunking, and Option 1 scoped RAG ingestion."""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from app.features.memory import ChatStore
from tests.astra_fake import InMemoryAstraCollection
from app.features.ocr import (
    DocumentValidationError,
    PageMarkdown,
    RenderedPage,
    clean_ocr_markdown,
    parse_pages_to_markdown,
    validate_upload_metadata,
)
from app.features.rag import (
    build_astra_documents,
    build_scope_filter,
    chunk_markdown_pages,
    delete_document_chunks,
    ingest_chunks,
    retrieve,
)
from app.features.rag.router import (
    delete_uploaded_document,
    get_rag_job_status,
    list_uploaded_documents,
    run_ingestion_pipeline,
    upload_rag_document,
)


class FakeInsertResult:
    def __init__(self, ids):
        self.inserted_ids = ids


class FakeDeleteResult:
    def __init__(self, count):
        self.deleted_count = count


class RecordingCollection:
    def __init__(self, fail_first_insert=False):
        self.inserted_batches = []
        self.find_filters = []
        self.delete_filters = []
        self.fail_first_insert = fail_first_insert

    def find(self, query_filter=None, **kwargs):
        self.find_filters.append(query_filter)
        return iter([{
            '_id': 'doc_1:0000',
            'text': 'Materi uji',
            'source_title': 'Modul 1',
            '$similarity': 0.92,
        }])

    def insert_many(self, docs, **kwargs):
        if self.fail_first_insert:
            self.fail_first_insert = False
            raise TimeoutError('Simulated transient network timeout')
        batch = list(docs)
        self.inserted_batches.append(batch)
        return FakeInsertResult([d['_id'] for d in batch])

    def delete_many(self, delete_filter, **kwargs):
        self.delete_filters.append(delete_filter)
        return FakeDeleteResult(3)


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


class FakeHttpResponse:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeUpload:
    def __init__(self, filename, content_bytes):
        self.filename = filename
        self._buf = io.BytesIO(content_bytes)

    async def read(self, size=-1):
        return self._buf.read(size)


def _make_request(app_state, headers=None):
    raw_headers = [
        (k.lower().encode('latin-1'), v.encode('latin-1'))
        for k, v in (headers or {}).items()
    ]
    scope = {
        'type': 'http',
        'method': 'GET',
        'path': '/',
        'headers': raw_headers,
        'query_string': b'',
        'app': type('FakeApp', (), {'state': app_state})(),
    }
    return Request(scope)


class OcrAndScopedRagTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ChatStore(collection=InMemoryAstraCollection())
        self.env_patch = patch.dict(os.environ, {
            'ASTRA_DB_API_ENDPOINT': 'https://example-astra.apps.astra.datastax.com',
            'ASTRA_DB_APPLICATION_TOKEN': 'AstraCS:test_token',
            'OCR_UPLOAD_DIR': str(Path(self.temp_dir.name) / 'uploads'),
        }, clear=False)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.temp_dir.cleanup()

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

        col = RecordingCollection()
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            retrieve('Apa itu overfitting?', user_id='siswa-1', retries=0, backoff=0)
        self.assertEqual(col.find_filters[0], student_filter)

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

        col = RecordingCollection(fail_first_insert=True)
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

    def test_upload_validation_and_end_to_end_pipeline_cleans_staging(self):
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('script.exe', 100)
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('kosong.pdf', 0)
        with self.assertRaises(DocumentValidationError):
            validate_upload_metadata('besar.pdf', 1024, max_bytes=512)

        staging = Path(self.temp_dir.name) / 'job_staging'
        staging.mkdir(parents=True, exist_ok=True)
        pdf_file = staging / 'materi.pdf'
        pdf_file.write_bytes(b'%PDF-1.4 fake')
        img_file = staging / 'p1.png'
        img_file.write_bytes(b'\x89PNG')

        job = self.store.create_ingestion_job(
            user_id='siswa-1',
            filename='materi.pdf',
            source_title='Materi Klasifikasi',
            topic='Klasifikasi',
            scope='user',
        )
        col = RecordingCollection()
        with patch(
            'app.features.rag.router.render_document_pages',
            return_value=[RenderedPage(1, img_file, '## Klasifikasi\n\nModel memetakan kelas diskret.')],
        ), patch.dict(os.environ, {'OCR_BACKEND': 'native_pdf'}), \
             patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            run_ingestion_pipeline(
                self.store,
                job_id=job['job_id'],
                doc_id=job['doc_id'],
                file_path=pdf_file,
                staging_dir=staging,
                user_id='siswa-1',
                filename='materi.pdf',
                source_title='Materi Klasifikasi',
                topic='Klasifikasi',
                scope='user',
            )

        updated = self.store.get_ingestion_job(job['job_id'])
        self.assertEqual(updated['status'], 'completed')
        self.assertEqual(updated['total_pages'], 1)
        self.assertEqual(updated['chunks_inserted'], 1)
        self.assertFalse(staging.exists())

    def test_router_enforces_global_admin_guard_and_user_ownership(self):
        state = type('AppState', (), {'store': self.store, 'assistant': None})()

        class FakeBgTasks:
            def __init__(self):
                self.tasks = []

            def add_task(self, fn, *args, **kwargs):
                self.tasks.append((fn, args, kwargs))

        async def call_upload(form_dict, headers=None):
            req = _make_request(state, headers=headers)

            async def fake_form():
                return form_dict

            req.form = fake_form
            bg = FakeBgTasks()
            resp = await upload_rag_document(req, bg)
            return resp, bg

        import asyncio

        with patch.dict(os.environ, {'RAG_ADMIN_SECRET': 'guru-rahasia'}):
            # Siswa biasa mencoba upload scope='global' -> harus ditolak 403
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(call_upload({
                    'file': FakeUpload('modul.pdf', b'%PDF-1.4 isi'),
                    'user_id': 'siswa-1',
                    'scope': 'global',
                }))
            self.assertEqual(ctx.exception.status_code, 403)

            # Guru dengan X-Admin-Key valid -> diterima 202
            resp_global, bg_global = asyncio.run(call_upload(
                {
                    'file': FakeUpload('modul_kelas.pdf', b'%PDF-1.4 isi'),
                    'user_id': 'guru-1',
                    'scope': 'global',
                },
                headers={'X-Admin-Key': 'guru-rahasia'},
            ))
            self.assertEqual(resp_global['scope'], 'global')
            self.assertEqual(len(bg_global.tasks), 1)
            # Bersihkan folder staging hasil test
            shutil.rmtree(bg_global.tasks[0][2]['staging_dir'], ignore_errors=True)

        # Siswa upload scope='user' -> diterima, dan Siswa-2 tidak boleh mengintip/menghapus
        resp_user, bg_user = asyncio.run(call_upload({
            'file': FakeUpload('catatan_siswa1.pdf', b'%PDF-1.4 isi'),
            'user_id': 'siswa-1',
            'scope': 'user',
        }))
        shutil.rmtree(bg_user.tasks[0][2]['staging_dir'], ignore_errors=True)

        req_view = _make_request(state)
        with self.assertRaises(HTTPException) as ctx_hide:
            get_rag_job_status(resp_user['job_id'], req_view, user_id='siswa-2')
        self.assertEqual(ctx_hide.exception.status_code, 404)

        own_docs = list_uploaded_documents('siswa-1', req_view)
        self.assertEqual(len(own_docs['documents']), 1)

        with self.assertRaises(HTTPException) as ctx_del:
            delete_uploaded_document(resp_user['doc_id'], req_view, user_id='siswa-2')
        self.assertEqual(ctx_del.exception.status_code, 403)

        col = RecordingCollection()
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            deleted = delete_uploaded_document(resp_user['doc_id'], req_view, user_id='siswa-1')
        self.assertTrue(deleted['deleted'])
        self.assertEqual(col.delete_filters[0], {
            'metadata.doc_id': resp_user['doc_id'],
            'metadata.scope': 'user',
            'metadata.uploaded_by': 'siswa-1',
        })


import shutil  # noqa: E402

if __name__ == '__main__':
    unittest.main()

