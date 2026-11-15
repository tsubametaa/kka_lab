"""Tests for app.features.rag.router (document upload, ingestion pipeline, scoped deletion)."""

import asyncio
import io
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.features.memory import ChatStore
from app.features.ocr import RenderedPage
from app.features.rag.router import (
    delete_uploaded_document,
    get_rag_job_status,
    list_uploaded_documents,
    run_ingestion_pipeline,
    upload_rag_document,
)
from tests.fakes import (
    FakeClient,
    InMemoryAstraCollection,
    RecordingAstraCollection,
    make_request,
)


class FakeUpload:
    def __init__(self, filename, content_bytes):
        self.filename = filename
        self._buf = io.BytesIO(content_bytes)

    async def read(self, size=-1):
        return self._buf.read(size)


class FakeBgTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append((fn, args, kwargs))


class RagRouterTests(unittest.TestCase):
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

    def test_end_to_end_ingestion_pipeline_cleans_staging(self):
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
        col = RecordingAstraCollection()
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
        state = SimpleNamespace(store=self.store, assistant=None)

        async def call_upload(form_dict, headers=None):
            req = make_request(app_state=state, headers=headers)

            async def fake_form():
                return form_dict

            req.form = fake_form
            bg = FakeBgTasks()
            resp = await upload_rag_document(req, bg)
            return resp, bg

        with patch.dict(os.environ, {'RAG_ADMIN_SECRET': 'guru-rahasia'}):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(call_upload({
                    'file': FakeUpload('modul.pdf', b'%PDF-1.4 isi'),
                    'user_id': 'siswa-1',
                    'scope': 'global',
                }))
            self.assertEqual(ctx.exception.status_code, 403)

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
            shutil.rmtree(bg_global.tasks[0][2]['staging_dir'], ignore_errors=True)

        resp_user, bg_user = asyncio.run(call_upload({
            'file': FakeUpload('catatan_siswa1.pdf', b'%PDF-1.4 isi'),
            'user_id': 'siswa-1',
            'scope': 'user',
        }))
        shutil.rmtree(bg_user.tasks[0][2]['staging_dir'], ignore_errors=True)

        req_view = make_request(app_state=state)
        with self.assertRaises(HTTPException) as ctx_hide:
            get_rag_job_status(resp_user['job_id'], req_view, user_id='siswa-2')
        self.assertEqual(ctx_hide.exception.status_code, 404)

        own_docs = list_uploaded_documents('siswa-1', req_view)
        self.assertEqual(len(own_docs['documents']), 1)

        with self.assertRaises(HTTPException) as ctx_del:
            delete_uploaded_document(resp_user['doc_id'], req_view, user_id='siswa-2')
        self.assertEqual(ctx_del.exception.status_code, 403)

        col = RecordingAstraCollection()
        with patch('app.features.rag.retriever.DataAPIClient', return_value=FakeClient(col)), \
             patch('app.features.rag.retriever.load_dotenv'):
            deleted = delete_uploaded_document(resp_user['doc_id'], req_view, user_id='siswa-1')
        self.assertTrue(deleted['deleted'])
        self.assertEqual(col.delete_filters[0], {
            'metadata.doc_id': resp_user['doc_id'],
            'metadata.scope': 'user',
            'metadata.uploaded_by': 'siswa-1',
        })


if __name__ == '__main__':
    unittest.main()
