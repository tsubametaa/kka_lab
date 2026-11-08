"""Astra DB vectorize document ingestion and scoped cleanup."""

import time
from datetime import datetime, timezone

from app.features.rag.exceptions import RagUnavailableError
from app.features.rag.retriever import (
    _http_status_code,
    _is_transient_error,
    get_astra_collection,
)

VALID_SCOPES = ('user', 'global')
DEFAULT_INSERT_BATCH = 20


def _call_with_astra_retry(fn, *, action_label, retries=2, backoff=1.5):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            status = _http_status_code(exc)
            if _is_transient_error(exc):
                if attempt < retries and backoff > 0:
                    time.sleep(backoff * (2 ** attempt))
                if attempt < retries:
                    continue
                if status == 503:
                    raise RagUnavailableError(
                        'Astra DB mengembalikan 503 Service Unavailable. '
                        'Database kemungkinan berstatus Hibernated/Resuming di '
                        'https://astra.datastax.com (klik Resume dan tunggu 1-2 menit).'
                    ) from exc
                raise RagUnavailableError(
                    f'Gagal {action_label} di Astra DB setelah {retries + 1} percobaan ({exc}).'
                ) from exc
            if status in (401, 403):
                raise RagUnavailableError(
                    f'Autentikasi Astra DB ditolak (HTTP {status}). '
                    'Periksa ASTRA_DB_APPLICATION_TOKEN pada .env.'
                ) from exc
            raise RagUnavailableError(
                f'Gagal {action_label} di Astra DB: {exc}'
            ) from exc


def build_astra_documents(chunks, *, doc_id, source_title, user_id, scope='user',
                          default_topic='Materi Tambahan', filename='',
                          created_at=None, extra_metadata=None):
    if scope not in VALID_SCOPES:
        raise ValueError(f"Scope harus salah satu dari {VALID_SCOPES}, bukan '{scope}'.")
    clean_user = (user_id or '').strip()
    if not clean_user:
        raise ValueError('user_id wajib diisi untuk pencatatan dokumen.')
    clean_doc_id = (doc_id or '').strip()
    if not clean_doc_id:
        raise ValueError('doc_id wajib diisi.')

    title = (source_title or filename or 'Materi KKA').strip()[:140]
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    docs = []
    for chunk in chunks:
        text = (chunk.text or '').strip()
        if not text:
            continue
        meta = {
            **(extra_metadata or {}),
            'doc_id': clean_doc_id,
            'scope': scope,
            'uploaded_by': clean_user,
            'chunk_index': chunk.index,
            'filename': (filename or '').strip()[:200],
            'created_at': timestamp,
        }
        docs.append({
            '_id': f'{clean_doc_id}:{chunk.index:04d}',
            'text': text,
            '$vectorize': text,
            'source_title': title,
            'topic': (chunk.topic or default_topic or 'Materi Tambahan').strip()[:140],
            'pages_pdf': chunk.pages_pdf,
            'metadata': meta,
        })
    return docs


def ingest_chunks(chunks, *, doc_id, source_title, user_id, scope='user',
                  default_topic='Materi Tambahan', filename='', extra_metadata=None,
                  batch_size=DEFAULT_INSERT_BATCH, retries=2, backoff=1.5):
    docs = build_astra_documents(
        chunks,
        doc_id=doc_id,
        source_title=source_title,
        user_id=user_id,
        scope=scope,
        default_topic=default_topic,
        filename=filename,
        extra_metadata=extra_metadata,
    )
    if not docs:
        return 0

    collection = get_astra_collection()
    step = max(1, int(batch_size))
    inserted = 0
    for start in range(0, len(docs), step):
        batch = docs[start:start + step]
        result = _call_with_astra_retry(
            lambda b=batch: collection.insert_many(b, ordered=False),
            action_label='menyimpan chunk materi',
            retries=retries,
            backoff=backoff,
        )
        inserted_ids = getattr(result, 'inserted_ids', None)
        inserted += len(inserted_ids) if inserted_ids is not None else len(batch)
    return inserted


def delete_document_chunks(doc_id, *, user_id=None, is_admin=False, retries=2, backoff=1.5):
    clean_doc_id = (doc_id or '').strip()
    if not clean_doc_id:
        raise ValueError('doc_id wajib diisi.')
    if is_admin:
        delete_filter = {'metadata.doc_id': clean_doc_id}
    else:
        clean_user = (user_id or '').strip()
        if not clean_user:
            raise ValueError('user_id wajib diisi untuk menghapus dokumen siswa.')
        delete_filter = {
            'metadata.doc_id': clean_doc_id,
            'metadata.scope': 'user',
            'metadata.uploaded_by': clean_user,
        }
    collection = get_astra_collection()
    result = _call_with_astra_retry(
        lambda: collection.delete_many(delete_filter),
        action_label='menghapus dokumen materi',
        retries=retries,
        backoff=backoff,
    )
    return getattr(result, 'deleted_count', 0)


def delete_user_rag_documents(user_id, *, retries=2, backoff=1.5):
    clean_user = (user_id or '').strip()
    if not clean_user:
        return 0
    collection = get_astra_collection()
    result = _call_with_astra_retry(
        lambda: collection.delete_many({
            'metadata.scope': 'user',
            'metadata.uploaded_by': clean_user,
        }),
        action_label='menghapus seluruh dokumen siswa',
        retries=retries,
        backoff=backoff,
    )
    return getattr(result, 'deleted_count', 0)

