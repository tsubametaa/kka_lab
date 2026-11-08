"""Astra DB vectorize client and multi-query retrieval."""

import os
import time

try:
    from astrapy import DataAPIClient
except ImportError:  # pragma: no cover
    DataAPIClient = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv():
        return False

from app.features.rag.exceptions import RagUnavailableError

TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}

def _http_status_code(exc):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    return status if isinstance(status, int) else None

def _is_transient_error(exc):
    status = _http_status_code(exc)
    if status in TRANSIENT_HTTP_STATUSES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return exc.__class__.__module__.startswith('httpx') and status is None

def build_scope_filter(user_id=None):
    """Filter Opsi 1: materi bawaan/global kelas + dokumen privat milik user_id."""
    branches = [
        {'metadata.scope': {'$exists': False}},
        {'metadata.scope': 'global'},
    ]
    clean_user = (user_id or '').strip()
    if clean_user:
        branches.append({
            'metadata.scope': 'user',
            'metadata.uploaded_by': clean_user,
        })
    return {'$or': branches}

def get_astra_collection():
    load_dotenv()
    endpoint = os.environ.get('ASTRA_DB_API_ENDPOINT', '').strip()
    token = os.environ.get('ASTRA_DB_APPLICATION_TOKEN', '').strip()
    if not endpoint or not token or endpoint.startswith('ISI_') or token.startswith('ISI_'):
        raise ValueError('Isi endpoint dan token asli pada .env.')
    if DataAPIClient is None:
        raise RuntimeError('Pustaka astrapy belum terpasang.')
    db = DataAPIClient().get_database(endpoint, token=token)
    return db.get_collection(
        os.getenv('ASTRA_DB_COLLECTION', 'rag_chatbot'),
        keyspace=os.getenv('ASTRA_DB_KEYSPACE', 'rag'),
    )

def _fetch_query_docs(collection, query, top_k, retries=2, backoff=1.5, query_filter=None):
    active_filter = query_filter if query_filter is not None else {}
    for attempt in range(retries + 1):
        try:
            cursor = collection.find(
                active_filter, sort={'$vectorize': query}, limit=top_k,
                projection={
                    '_id': True, 'text': True, 'source_title': True,
                    'topic': True, 'pages_pdf': True, 'metadata': True,
                },
                include_similarity=True,
            )
            return list(cursor)
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
                    f'Gagal menghubungi Astra DB setelah {retries + 1} percobaan ({exc}).'
                ) from exc
            if status in (401, 403):
                raise RagUnavailableError(
                    f'Autentikasi Astra DB ditolak (HTTP {status}). '
                    'Periksa ASTRA_DB_APPLICATION_TOKEN pada .env.'
                ) from exc
            raise RagUnavailableError(
                f'Gagal mengambil konteks dari Astra DB: {exc}'
            ) from exc

def retrieve(question, top_k=3, retries=2, backoff=1.5, user_id=None):
    collection = get_astra_collection()
    scope_filter = build_scope_filter(user_id)
    queries = [question] if isinstance(question, str) else question
    queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
    documents = {}
    for query in queries:
        if len(query) > 1000:
            search_text = f'{query[:500].rstrip()}\n{query[-500:].lstrip()}'
        else:
            search_text = query.strip()
        results = _fetch_query_docs(
            collection, search_text, top_k, retries=retries, backoff=backoff,
            query_filter=scope_filter,
        )
        for doc in results:
            key = str(doc.get('_id') or (doc.get('source_title'), doc.get('text')))
            current = documents.get(key)
            similarity = doc.get('$similarity')
            current_similarity = current.get('$similarity') if current else None
            if current is None or (
                isinstance(similarity, (int, float))
                and (not isinstance(current_similarity, (int, float))
                     or similarity > current_similarity)
            ):
                documents[key] = doc
    ranked = sorted(
        documents.values(),
        key=lambda doc: doc.get('$similarity')
        if isinstance(doc.get('$similarity'), (int, float)) else -1,
        reverse=True,
    )
    return ranked[:top_k]

