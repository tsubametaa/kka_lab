"""Astra DB vectorize document ingestion and scoped cleanup."""

import re
import time
from datetime import datetime, timezone

from app.features.rag.chunker import _estimate_embed_tokens
from app.features.rag.exceptions import RagUnavailableError
from app.features.rag.retriever import (
    _http_status_code,
    _is_transient_error,
    get_astra_collection,
)

VALID_SCOPES = ('user', 'global')
DEFAULT_INSERT_BATCH = 20
MAX_VECTORIZE_CHARS = 1100
MAX_VECTORIZE_EST_TOKENS = 340


def _split_text_for_embedding(text, *, max_chars=MAX_VECTORIZE_CHARS, max_est_tokens=MAX_VECTORIZE_EST_TOKENS):
    """Ensure every document's $vectorize payload safely fits within NVIDIA E5 512-token cap."""
    clean = (text or '').strip()
    if not clean:
        return []
    if len(clean) <= max_chars and _estimate_embed_tokens(clean) <= max_est_tokens:
        return [clean]

    words = clean.split()
    parts = []
    cur = []
    for word in words:
        # Handle ultra-long unbroken tokens (e.g. URLs/base64).
        if len(word) > max_chars or _estimate_embed_tokens(word) > max_est_tokens:
            if cur:
                parts.append(' '.join(cur))
                cur = []
            step = max(80, min(max_chars // 2, max_est_tokens * 2))
            for i in range(0, len(word), step):
                parts.append(word[i:i + step])
            continue
        candidate = ' '.join([*cur, word]) if cur else word
        if cur and (len(candidate) > max_chars or _estimate_embed_tokens(candidate) > max_est_tokens):
            parts.append(' '.join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        parts.append(' '.join(cur))
    return parts or [clean[:max_chars]]


def _is_embedding_token_limit_error(exc):
    msg = str(exc).lower()
    return 'exceeds maximum allowed token size' in msg or 'token size 512' in msg


def _halve_doc(doc):
    text = (doc.get('$vectorize') or doc.get('text') or '').strip()
    words = text.split()
    if len(words) >= 2:
        mid = len(words) // 2
        halves = [' '.join(words[:mid]).strip(), ' '.join(words[mid:]).strip()]
    elif len(text) >= 2:
        mid = len(text) // 2
        halves = [text[:mid].strip(), text[mid:].strip()]
    else:
        return [doc]

    out = []
    for sub_idx, part in enumerate(h for h in halves if h):
        cloned = dict(doc)
        cloned['_id'] = f"{doc['_id']}_s{sub_idx}"
        cloned['text'] = part
        cloned['$vectorize'] = part
        out.append(cloned)
    return out or [doc]


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
        pieces = _split_text_for_embedding(text)
        for sub_idx, piece in enumerate(pieces):
            doc_key = (
                f'{clean_doc_id}:{chunk.index:04d}'
                if len(pieces) == 1
                else f'{clean_doc_id}:{chunk.index:04d}_{sub_idx:02d}'
            )
            meta = {
                **(extra_metadata or {}),
                'doc_id': clean_doc_id,
                'scope': scope,
                'uploaded_by': clean_user,
                'chunk_index': chunk.index,
                'sub_index': sub_idx,
                'filename': (filename or '').strip()[:200],
                'created_at': timestamp,
            }
            docs.append({
                '_id': doc_key,
                'text': piece,
                '$vectorize': piece,
                'source_title': title,
                'topic': (chunk.topic or default_topic or 'Materi Tambahan').strip()[:140],
                'pages_pdf': chunk.pages_pdf,
                'metadata': meta,
            })
    return docs


def _insert_batch_resilient(collection, batch, *, retries, backoff, depth=0):
    """Insert a batch into Astra DB, auto-splitting docs if provider rejects >512 tokens."""
    try:
        result = _call_with_astra_retry(
            lambda b=batch: collection.insert_many(b, ordered=True),
            action_label='menyimpan chunk materi',
            retries=retries,
            backoff=backoff,
        )
        inserted_ids = getattr(result, 'inserted_ids', None)
        return len(inserted_ids) if inserted_ids is not None else len(batch)
    except RagUnavailableError as exc:
        if not _is_embedding_token_limit_error(exc) or depth >= 4:
            raise
        if len(batch) > 1:
            total = 0
            for item in batch:
                total += _insert_batch_resilient(
                    collection, [item], retries=retries, backoff=backoff, depth=depth
                )
            return total
        split_docs = _halve_doc(batch[0])
        if len(split_docs) <= 1:
            raise
        total = 0
        for item in split_docs:
            total += _insert_batch_resilient(
                collection, [item], retries=retries, backoff=backoff, depth=depth + 1
            )
        return total


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
    try:
        for start in range(0, len(docs), step):
            batch = docs[start:start + step]
            inserted += _insert_batch_resilient(
                collection, batch, retries=retries, backoff=backoff
            )
    except Exception:
        if inserted > 0:
            try:
                delete_document_chunks(doc_id, user_id=user_id, is_admin=(scope == 'global'), retries=1, backoff=0)
            except Exception:
                pass
        raise
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

