"""Astra DB vectorize client and multi-query retrieval."""

import os
import re
import threading
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

from app.features.language.slang import normalize_slang
from app.features.rag.exceptions import RagUnavailableError

TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
DEFAULT_MIN_SIMILARITY = 0.70

_WORD_RE = re.compile(r'[a-z0-9]+')
_STOPWORDS = frozenset({
    'apa', 'apakah', 'bagaimana', 'gimana', 'mengapa', 'kenapa', 'siapa', 'kapan',
    'dimana', 'yang', 'dan', 'atau', 'itu', 'ini', 'adalah', 'untuk', 'dengan',
    'dari', 'pada', 'ke', 'di', 'dalam', 'tentang', 'jelaskan', 'sebutkan',
    'buatkan', 'berikan', 'tolong', 'coba', 'saya', 'aku', 'kamu', 'pertanyaan',
    'soal', 'contoh', 'contohnya', 'materi', 'pelajaran', 'kelas', 'bab', 'smk',
    'sma', 'seputar', 'mengenai', 'pilihan', 'ganda', 'kuis', 'quiz', 'belajar',
    'what', 'how', 'why', 'is', 'are', 'the', 'a', 'an', 'of', 'in', 'on', 'to',
    'for', 'with', 'about', 'and', 'or', 'please', 'explain', 'give', 'me',
})

# Sinyal eksplisit bahwa siswa memang bertanya soal pemrograman / koding / sintaks / algoritma.
_PROGRAMMING_QUERY_RE = re.compile(
    r'\b(?:python|pemrograman|programming|koding|coding|kode|code|sintaks|syntax|'
    r'algoritma|algorithm|flowchart|pseudocode|variabel|variable|tipe\s+data|data\s+type|'
    r'percabangan|kondisional|conditional|perulangan|loop|looping|for\s+loop|while\s+loop|'
    r'fungsi|function|debug(?:ging)?|kompilasi|compiler|interpreter|array|tuple|dictionary|'
    r'operator\s+aritmatika|rekursi|recursion|sorting|searching|sql|basis\s+data|database)\b'
    r'|input\(|print\(|>=|<=|==|!=',
    flags=re.IGNORECASE,
)

# Penanda dokumen yang topik/judulnya secara khusus membahas materi pemrograman/koding.
_PROGRAMMING_TOPIC_RE = re.compile(
    r'\b(?:pemrograman|python|debugging|kode\s+program|sintaks\s+dasar|flowchart|pseudocode|'
    r'looping|pengulangan|percabangan|variabel|tipe\s+data|operator\s+aritmatika|'
    r'algoritma\s+dasar|algoritma\s+terstruktur|algoritma\s+dan\s+pemrograman|pemrograman\s+teks)\b',
    flags=re.IGNORECASE,
)

# Pengecualian: dokumen yang membandingkan pemrograman biasa dengan AI/ML tetap relevan untuk pertanyaan AI.
_AI_BRIDGE_TOPIC_RE = re.compile(
    r'\b(?:machine\s+learning|kecerdasan\s+artifisial|ai\b|ka\b|konsep\s+koding\s+dan\s+pemrograman)\b',
    flags=re.IGNORECASE,
)

# Penanda chunk meta/pembuka bab yang kurang informatif untuk pertanyaan konsep/latihan.
_BOILERPLATE_TOPIC_RE = re.compile(
    r'\b(?:orientasi,\s*tujuan,\s*apersepsi|formatif\s+awal|daftar\s+isi|kata\s+pengantar)\b',
    flags=re.IGNORECASE,
)


def get_min_similarity(min_similarity=None):
    """Return configured minimum vector similarity floor (<= 0 disables filtering)."""
    if min_similarity is not None:
        return float(min_similarity)
    raw = os.getenv('RAG_MIN_SIMILARITY', str(DEFAULT_MIN_SIMILARITY)).strip()
    if not raw:
        return DEFAULT_MIN_SIMILARITY
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_MIN_SIMILARITY

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

_ASTRA_COLLECTION_CACHE = {}
_ASTRA_COLLECTION_LOCK = threading.Lock()


def get_astra_collection():
    load_dotenv()
    endpoint = os.environ.get('ASTRA_DB_API_ENDPOINT', '').strip()
    token = os.environ.get('ASTRA_DB_APPLICATION_TOKEN', '').strip()
    if not endpoint or not token or endpoint.startswith('ISI_') or token.startswith('ISI_'):
        raise ValueError('Isi endpoint dan token asli pada .env.')
    if DataAPIClient is None:
        raise RuntimeError('Pustaka astrapy belum terpasang.')
    col_name = os.getenv('ASTRA_DB_COLLECTION', 'rag_chatbot')
    keyspace = os.getenv('ASTRA_DB_KEYSPACE', 'rag')
    # Reuse persistent HTTP/2 connection pool in production; skip caching when mocked in unit tests.
    is_real_client = isinstance(DataAPIClient, type) and DataAPIClient.__module__.startswith('astrapy')
    if not is_real_client:
        db = DataAPIClient().get_database(endpoint, token=token)
        return db.get_collection(col_name, keyspace=keyspace)

    cache_key = (endpoint, token, col_name, keyspace)
    cached = _ASTRA_COLLECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _ASTRA_COLLECTION_LOCK:
        cached = _ASTRA_COLLECTION_CACHE.get(cache_key)
        if cached is None:
            db = DataAPIClient().get_database(endpoint, token=token)
            cached = db.get_collection(col_name, keyspace=keyspace)
            _ASTRA_COLLECTION_CACHE[cache_key] = cached
    return cached

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

def _condense_queries(question):
    queries = [question] if isinstance(question, str) else question
    cleaned = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
    condensed = []
    for q in cleaned:
        if len(q) > 1000:
            condensed.append(f'{q[:500].rstrip()}\n{q[-500:].lstrip()}')
        else:
            condensed.append(q)
    return condensed


def _extract_terms(text):
    expanded = normalize_slang(text or '').lower()
    return {w for w in _WORD_RE.findall(expanded) if len(w) >= 2 and w not in _STOPWORDS}


def _is_programming_query(queries):
    combined = ' '.join(normalize_slang(q or '') for q in (queries or ()))
    return bool(_PROGRAMMING_QUERY_RE.search(combined))


def _is_programming_doc(doc):
    meta = doc.get('metadata') or {}
    # Jangan pernah menghukum dokumen privat yang diunggah langsung oleh siswa.
    if meta.get('scope') == 'user':
        return False
    topic = str(doc.get('topic') or meta.get('topic') or '').strip()
    if not topic:
        first_line = str(doc.get('text') or '').splitlines()[0] if doc.get('text') else ''
        if first_line.lower().startswith('topik:'):
            topic = first_line
    if not topic:
        return False
    if _AI_BRIDGE_TOPIC_RE.search(topic):
        return False
    return bool(_PROGRAMMING_TOPIC_RE.search(topic))


def _is_boilerplate_doc(doc):
    meta = doc.get('metadata') or {}
    if meta.get('scope') == 'user':
        return False
    topic = str(doc.get('topic') or meta.get('topic') or '').strip()
    return bool(topic and _BOILERPLATE_TOPIC_RE.search(topic))


def _hybrid_rerank_score(doc, query_terms, wants_programming, wants_boilerplate):
    sim = doc.get('$similarity')
    base = float(sim) if isinstance(sim, (int, float)) else -1.0
    if not query_terms:
        return base

    meta = doc.get('metadata') or {}
    topic = str(doc.get('topic') or meta.get('topic') or '')
    title = str(doc.get('source_title') or meta.get('source_title') or '')
    body = str(doc.get('text') or '')[:600]

    topic_terms = _extract_terms(f'{topic} {title}')
    body_terms = _extract_terms(body)

    topic_overlap = len(query_terms & topic_terms) / len(query_terms) if topic_terms else 0.0
    body_overlap = len(query_terms & body_terms) / len(query_terms) if body_terms else 0.0

    score = base + (0.07 * topic_overlap) + (0.03 * body_overlap)

    if _is_programming_doc(doc) and not wants_programming:
        score -= 0.14
    if _is_boilerplate_doc(doc) and not wants_boilerplate:
        score -= 0.06
    return score


def _rank_and_filter_docs(doc_batches, *, min_similarity=None, queries=None):
    documents = {}
    for results in doc_batches:
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

    threshold = get_min_similarity(min_similarity)
    candidates = list(documents.values())
    if threshold > 0:
        candidates = [
            doc for doc in candidates
            if not isinstance(doc.get('$similarity'), (int, float))
            or doc['$similarity'] >= threshold
        ]

    if not queries:
        return sorted(
            candidates,
            key=lambda doc: doc.get('$similarity')
            if isinstance(doc.get('$similarity'), (int, float)) else -1,
            reverse=True,
        )

    query_list = [queries] if isinstance(queries, str) else list(queries)
    query_terms = _extract_terms(' '.join(query_list))
    wants_programming = _is_programming_query(query_list)
    wants_boilerplate = any(
        kw in ' '.join(query_list).lower() for kw in ('apersepsi', 'tujuan pembelajaran', 'orientasi')
    )

    # Jika pertanyaan siswa bukan tentang pemrograman dan tersedia dokumen non-pemrograman yang lolos,
    # sisihkan dokumen murni pemrograman agar tidak mencemari konteks dan label topik.
    if not wants_programming:
        non_prog = [doc for doc in candidates if not _is_programming_doc(doc)]
        if non_prog:
            candidates = non_prog

    return sorted(
        candidates,
        key=lambda doc: _hybrid_rerank_score(doc, query_terms, wants_programming, wants_boilerplate),
        reverse=True,
    )


def retrieve(question, top_k=3, retries=2, backoff=1.5, user_id=None, min_similarity=None):
    collection = get_astra_collection()
    scope_filter = build_scope_filter(user_id)
    queries = _condense_queries(question)
    fetch_k = min(max(int(top_k) * 3, 9), 15)
    batches = [
        _fetch_query_docs(
            collection, search_text, fetch_k, retries=retries, backoff=backoff,
            query_filter=scope_filter,
        )
        for search_text in queries
    ]
    return _rank_and_filter_docs(
        batches, min_similarity=min_similarity, queries=queries
    )[:top_k]


def fetch_user_document_chunks(user_id, *, doc_id=None, limit=8, retries=2, backoff=1.5):
    """Fetch chunks of a student's uploaded document in reading order (chunk_index)."""
    clean_user = (user_id or '').strip()
    if not clean_user:
        return []
    collection = get_astra_collection()
    filt = {'metadata.scope': 'user', 'metadata.uploaded_by': clean_user}
    if doc_id:
        filt['metadata.doc_id'] = str(doc_id).strip()

    for attempt in range(retries + 1):
        try:
            cursor = collection.find(
                filt,
                limit=max(20, int(limit) * 3),
                projection={
                    '_id': True, 'text': True, 'source_title': True,
                    'topic': True, 'pages_pdf': True, 'metadata': True,
                },
            )
            docs = list(cursor)
            break
        except Exception as exc:
            if _is_transient_error(exc) and attempt < retries:
                if backoff > 0:
                    time.sleep(backoff * (2 ** attempt))
                continue
            return []

    docs.sort(
        key=lambda d: (
            (d.get('metadata') or {}).get('chunk_index', 0),
            (d.get('metadata') or {}).get('sub_index', 0),
            str(d.get('_id', '')),
        )
    )
    return docs[:limit]


def retrieve_for_chat(question, *, user_id=None, latest_doc_id=None, explicit_doc_ref=False,
                      top_k=5, user_top_k=5, retries=2, backoff=1.5, min_similarity=None):
    """Retrieve user-uploaded documents with priority alongside global KKA material."""
    clean_user = (user_id or '').strip()
    if not clean_user:
        return retrieve(
            question, top_k=top_k, retries=retries, backoff=backoff, min_similarity=min_similarity
        )

    collection = get_astra_collection()
    queries = _condense_queries(question)
    threshold = get_min_similarity(min_similarity)

    user_filter = {'metadata.scope': 'user', 'metadata.uploaded_by': clean_user}
    if latest_doc_id and explicit_doc_ref:
        user_filter['metadata.doc_id'] = str(latest_doc_id).strip()

    user_batches = [
        _fetch_query_docs(
            collection, q, max(user_top_k, 4), retries=retries, backoff=backoff,
            query_filter=user_filter,
        )
        for q in queries
    ]
    user_floor = 0.0 if explicit_doc_ref else (min(threshold, 0.48) if threshold > 0 else 0.0)
    user_hits = _rank_and_filter_docs(user_batches, min_similarity=user_floor)[:user_top_k]

    if explicit_doc_ref or len(user_hits) < 2:
        ordered_docs = fetch_user_document_chunks(
            clean_user, doc_id=latest_doc_id, limit=max(user_top_k, 6),
            retries=retries, backoff=backoff,
        )
        seen_ids = {str(d.get('_id', '')) for d in user_hits}
        for doc in ordered_docs:
            key = str(doc.get('_id', ''))
            if key not in seen_ids:
                user_hits.append(doc)
                seen_ids.add(key)
            if len(user_hits) >= user_top_k:
                break

    if explicit_doc_ref and user_hits:
        return user_hits[:max(top_k, user_top_k)]

    global_filter = build_scope_filter(None)
    global_limit = min(3, top_k)
    global_fetch_k = min(max(global_limit * 3, 9), 15)
    global_batches = [
        _fetch_query_docs(
            collection, q, global_fetch_k, retries=retries, backoff=backoff,
            query_filter=global_filter,
        )
        for q in queries
    ]
    global_hits = _rank_and_filter_docs(
        global_batches, min_similarity=min_similarity, queries=queries
    )[:global_limit]

    merged = []
    seen = set()
    for doc in [*user_hits, *global_hits]:
        key = str(doc.get('_id') or (doc.get('source_title'), doc.get('text')))
        if key not in seen:
            merged.append(doc)
            seen.add(key)
    return merged[:max(top_k, len(user_hits) + 1)]

