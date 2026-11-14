"""FastAPI endpoints for Unlimited-OCR document upload, job polling, and scoped RAG management."""

import logging
import os
import secrets
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.core.security import verify_internal_api_key as _verify_internal_api_key

from app.features.ocr import (
    DocumentValidationError,
    OcrUnavailableError,
    max_upload_bytes,
    parse_pages_to_markdown,
    render_document_pages,
    validate_upload_metadata,
)
from app.features.rag.chunker import chunk_markdown_pages
from app.features.rag.exceptions import RagUnavailableError
from app.features.rag.ingestor import (
    VALID_SCOPES,
    delete_document_chunks,
    ingest_chunks,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=['rag'])

_UPLOAD_OPENAPI = {
    'requestBody': {
        'required': True,
        'content': {
            'multipart/form-data': {
                'schema': {
                    'type': 'object',
                    'required': ['file', 'user_id'],
                    'properties': {
                        'file': {'type': 'string', 'format': 'binary'},
                        'user_id': {'type': 'string', 'minLength': 1, 'maxLength': 128},
                        'scope': {'type': 'string', 'enum': ['user', 'global'], 'default': 'user'},
                        'source_title': {'type': 'string', 'maxLength': 140},
                        'topic': {'type': 'string', 'maxLength': 140},
                    },
                },
            },
        },
    },
}

def _is_admin_request(request: Request):
    admin_secret = os.getenv('RAG_ADMIN_SECRET', '').strip()
    if not admin_secret:
        return False
    provided = request.headers.get('x-admin-key', '')
    return secrets.compare_digest(provided, admin_secret)

def _stage_upload_file(staging_dir: Path, staged_file: Path, raw_bytes: bytes):
    """Tulis berkas unggahan ke direktori staging (sinkron, dipanggil via threadpool)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_file.write_bytes(raw_bytes)


def run_ingestion_pipeline(store, *, job_id, doc_id, file_path, staging_dir,
                           user_id, filename, source_title, topic, scope, tokenizer=None):
    """Fungsi sinkron (dijalankan di threadpool BackgroundTasks) untuk OCR -> Chunk -> Astra DB."""
    try:
        store.update_ingestion_job(job_id, status='processing')
        rendered_pages = render_document_pages(file_path, Path(staging_dir) / 'pages')
        markdown_pages = parse_pages_to_markdown(rendered_pages, work_dir=staging_dir)
        chunks = chunk_markdown_pages(
            markdown_pages,
            tokenizer=tokenizer,
            default_topic=topic or 'Materi Tambahan',
        )
        if not chunks:
            raise DocumentValidationError('Dokumen tidak menghasilkan potongan teks materi.')
        inserted = ingest_chunks(
            chunks,
            doc_id=doc_id,
            source_title=source_title,
            user_id=user_id,
            scope=scope,
            default_topic=topic or 'Materi Tambahan',
            filename=filename,
        )
        store.update_ingestion_job(
            job_id,
            status='completed',
            total_pages=len(rendered_pages),
            chunks_inserted=inserted,
            error_message='',
        )
    except (DocumentValidationError, OcrUnavailableError, RagUnavailableError, ValueError) as exc:
        logger.warning('Job OCR/RAG %s gagal: %s', job_id, exc)
        store.update_ingestion_job(job_id, status='failed', error_message=str(exc))
    except Exception as exc:
        logger.exception('Kesalahan tidak terduga pada job OCR/RAG %s', job_id)
        store.update_ingestion_job(
            job_id, status='failed', error_message=f'Kesalahan internal pemrosesan dokumen: {exc}'
        )
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


@router.post('/api/rag/documents', status_code=202, openapi_extra=_UPLOAD_OPENAPI)
async def upload_rag_document(request: Request, background_tasks: BackgroundTasks):
    _verify_internal_api_key(request)
    try:
        form = await request.form()
    except (AssertionError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Paket 'python-multipart' belum terpasang. Jalankan: pip install -r requirements-ocr.txt",
        ) from exc

    upload = form.get('file')
    if upload is None or not hasattr(upload, 'read'):
        raise HTTPException(status_code=422, detail="Field berkas 'file' wajib disertakan.")

    user_id = str(form.get('user_id') or '').strip()
    if not user_id or len(user_id) > 128:
        raise HTTPException(status_code=422, detail='user_id wajib diisi (1-128 karakter).')

    scope = str(form.get('scope') or 'user').strip().lower()
    if scope not in VALID_SCOPES:
        raise HTTPException(status_code=422, detail="Scope harus 'user' atau 'global'.")
    if scope == 'global' and not _is_admin_request(request):
        raise HTTPException(
            status_code=403,
            detail='Pengunggahan materi global kelas memerlukan izin guru/admin (X-Admin-Key).',
        )

    limit = max_upload_bytes()
    raw_bytes = await upload.read(limit + 1)
    try:
        clean_name, _ = validate_upload_metadata(
            getattr(upload, 'filename', ''), len(raw_bytes), max_bytes=limit
        )
    except DocumentValidationError as exc:
        status = 413 if len(raw_bytes) > limit else 422
        raise HTTPException(status_code=status, detail=str(exc)) from exc

    raw_title = str(form.get('source_title') or '').strip()
    source_title = (raw_title or Path(clean_name).stem or clean_name)[:140]
    topic = str(form.get('topic') or 'Materi Tambahan').strip()[:140]

    job_id = str(uuid.uuid4())
    doc_id = f'doc_{uuid.uuid4().hex[:16]}'
    upload_root = Path(os.getenv('OCR_UPLOAD_DIR', 'data/uploads'))
    staging_dir = upload_root / job_id
    staged_file = staging_dir / clean_name
    await run_in_threadpool(_stage_upload_file, staging_dir, staged_file, raw_bytes)

    store = request.app.state.store
    try:
        job = await run_in_threadpool(
            store.create_ingestion_job,
            job_id=job_id,
            doc_id=doc_id,
            user_id=user_id,
            filename=clean_name,
            source_title=source_title,
            topic=topic,
            scope=scope,
        )
    except Exception:
        await run_in_threadpool(shutil.rmtree, staging_dir, ignore_errors=True)
        raise

    assistant = getattr(request.app.state, 'assistant', None)
    tokenizer = getattr(assistant, 'tokenizer', None)

    background_tasks.add_task(
        run_ingestion_pipeline,
        store,
        job_id=job_id,
        doc_id=doc_id,
        file_path=staged_file,
        staging_dir=staging_dir,
        user_id=user_id,
        filename=clean_name,
        source_title=source_title,
        topic=topic,
        scope=scope,
        tokenizer=tokenizer,
    )
    return {
        'job_id': job['job_id'],
        'doc_id': job['doc_id'],
        'status': job['status'],
        'scope': job['scope'],
        'source_title': job['source_title'],
    }


@router.get('/api/rag/jobs/{job_id}')
def get_rag_job_status(job_id: str, request: Request, user_id: str = Query(default='')):
    _verify_internal_api_key(request)
    job = request.app.state.store.get_ingestion_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job pemrosesan dokumen tidak ditemukan.')
    clean_user = user_id.strip() if isinstance(user_id, str) else ''
    if (
        clean_user
        and job['scope'] == 'user'
        and job['user_id'] != clean_user
        and not _is_admin_request(request)
    ):
        raise HTTPException(status_code=404, detail='Job pemrosesan dokumen tidak ditemukan.')
    return job


@router.get('/api/users/{user_id}/documents')
def list_uploaded_documents(user_id: str, request: Request):
    _verify_internal_api_key(request)
    return {'documents': request.app.state.store.list_user_documents(user_id)}


@router.get('/api/rag/auto-learned')
def list_auto_learned_kka_notes(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
):
    _verify_internal_api_key(request)
    if os.getenv('RAG_ADMIN_SECRET', '').strip() and not _is_admin_request(request):
        raise HTTPException(
            status_code=403,
            detail='Melihat daftar materi hasil kurasi otomatis memerlukan izin guru/admin (X-Admin-Key).',
        )
    max_items = limit if isinstance(limit, int) else 50
    return {'items': request.app.state.store.list_auto_learned_records(limit=max_items)}


@router.delete('/api/rag/documents/{doc_id}')
def delete_uploaded_document(
    doc_id: str,
    request: Request,
    user_id: str = Query(default=''),
):
    _verify_internal_api_key(request)
    is_admin = _is_admin_request(request)
    store = request.app.state.store
    record = store.get_document_record(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail='Dokumen tidak ditemukan.')

    clean_user = user_id.strip() if isinstance(user_id, str) else ''
    if record['scope'] == 'global' and not is_admin:
        raise HTTPException(
            status_code=403,
            detail='Penghapusan materi global kelas memerlukan izin guru/admin (X-Admin-Key).',
        )
    if not is_admin and (not clean_user or record['user_id'] != clean_user):
        raise HTTPException(status_code=403, detail='Anda tidak berhak menghapus dokumen ini.')

    try:
        deleted_chunks = delete_document_chunks(
            doc_id, user_id=clean_user, is_admin=is_admin
        )
    except RagUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    store.delete_document_record(doc_id, user_id=clean_user, is_admin=is_admin)
    return {'deleted': True, 'doc_id': doc_id, 'deleted_chunks': deleted_chunks}
