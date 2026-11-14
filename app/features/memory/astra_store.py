"""Astra DB-backed NoSQL conversation, memory, and RAG metadata store."""

from datetime import datetime, timezone
import hashlib
import logging
import os
import threading
import time
import uuid

try:
    from astrapy import DataAPIClient
except ImportError:  # pragma: no cover
    DataAPIClient = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv():
        return False

from app.features.memory.gate import (
    _knowledge_terms,
    normalize_question,
    training_quality,
)

logger = logging.getLogger(__name__)
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}


def _http_status_code(exc):
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    return status if isinstance(status, int) else None


def _is_transient_error(exc):
    status = _http_status_code(exc)
    if status in TRANSIENT_HTTP_STATUSES or isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    return exc.__class__.__module__.startswith('httpx') and status is None


def _call_with_retry(fn, *, retries=2, backoff=1.0):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if _is_transient_error(exc) and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise


def get_astra_chat_collection(collection_name=None, keyspace=None):
    """Ambil atau buat koleksi NoSQL chat_history di Astra DB."""
    load_dotenv()
    endpoint = os.environ.get('ASTRA_DB_API_ENDPOINT', '').strip()
    token = os.environ.get('ASTRA_DB_APPLICATION_TOKEN', '').strip()
    if not endpoint or not token or endpoint.startswith('ISI_') or token.startswith('ISI_'):
        raise ValueError('Isi ASTRA_DB_API_ENDPOINT dan ASTRA_DB_APPLICATION_TOKEN pada .env.')
    if DataAPIClient is None:
        raise RuntimeError('Pustaka astrapy belum terpasang. Jalankan pip install astrapy.')

    col_name = collection_name or os.getenv('ASTRA_DB_CHAT_COLLECTION', 'chat_history').strip()
    ks = keyspace or os.getenv('ASTRA_DB_KEYSPACE', 'rag').strip()
    db = DataAPIClient().get_database(endpoint, token=token)
    try:
        return db.create_collection(col_name, keyspace=ks)
    except Exception:
        return db.get_collection(col_name, keyspace=ks)


class AstraChatStore:
    """Implementasi penyimpanan NoSQL Astra DB untuk sesi, memori, persona, & metadata RAG."""

    def __init__(self, collection=None, collection_name=None, keyspace=None):
        self._collection = collection
        self._collection_name = collection_name
        self._keyspace = keyspace
        self._collection_lock = threading.Lock()

    @property
    def collection(self):
        if self._collection is None:
            with self._collection_lock:
                if self._collection is None:
                    self._collection = get_astra_chat_collection(
                        self._collection_name, self._keyspace
                    )
        return self._collection

    @staticmethod
    def _clean_doc(row):
        if row is None:
            return None
        return {k: v for k, v in row.items() if k not in ('_id', 'type', 'timestamp')}

    def _upsert_doc(self, doc_id, fields):
        _call_with_retry(
            lambda: self.collection.update_one({'_id': doc_id}, {'$set': fields}, upsert=True)
        )

    # --- Sesi & Riwayat Chat ---

    def create_session(self, user_id):
        session_id = str(uuid.uuid4())
        now_dt = datetime.now(timezone.utc)
        doc = {
            '_id': f'sess_{session_id}',
            'type': 'session',
            'session_id': session_id,
            'user_id': user_id,
            'created_at': now_dt.isoformat(),
            'timestamp': now_dt.timestamp(),
        }
        _call_with_retry(lambda: self.collection.insert_one(doc))
        return session_id

    def session_belongs_to(self, session_id, user_id):
        found = _call_with_retry(
            lambda: self.collection.find_one({'session_id': session_id, 'user_id': user_id})
        )
        return found is not None

    def history(self, session_id, limit=4):
        safe_limit = max(int(limit), 0)
        if safe_limit == 0:
            return []
        docs = _call_with_retry(lambda: list(self.collection.find(
            {'session_id': session_id, 'type': 'exchange'},
            projection={'question': True, 'answer': True, 'created_at': True, 'timestamp': True},
            sort={'timestamp': -1},
            limit=safe_limit,
        )))
        docs.sort(key=lambda d: d.get('timestamp') or d.get('created_at') or '')
        return [
            msg
            for row in docs[-safe_limit:]
            for msg in (
                {'role': 'user', 'content': row.get('question', '')},
                {'role': 'assistant', 'content': row.get('answer', '')},
            )
        ]

    def save_exchange(self, user_id, session_id, question, answer, mode,
                      sources, training_consent, memory_replay=False):
        eligible, reason = training_quality(question, answer, training_consent)
        if memory_replay:
            eligible, reason = False, 'answer_reused_from_memory'
        status = 'pending_review' if eligible else 'excluded'
        now_dt = datetime.now(timezone.utc)
        doc = {
            '_id': f'ex_{uuid.uuid4().hex}',
            'type': 'exchange',
            'user_id': user_id,
            'session_id': session_id,
            'question': question,
            'normalized_question': normalize_question(question),
            'answer': answer,
            'mode': mode,
            'sources': sources,
            'training_status': status,
            'training_reason': reason,
            'created_at': now_dt.isoformat(),
            'timestamp': now_dt.timestamp(),
        }
        _call_with_retry(lambda: self.collection.insert_one(doc))
        return {'eligible_for_review': eligible, 'status': status, 'reason': reason}

    def find_repeated_question(self, user_id, question, mode=None):
        normalized = normalize_question(question)
        if len(normalized.split()) < 3:
            return None
        query = {'type': 'exchange', 'user_id': user_id, 'normalized_question': normalized}
        if mode:
            query['mode'] = mode
        docs = _call_with_retry(lambda: list(self.collection.find(
            query,
            projection={'answer': True, 'sources': True, 'created_at': True, 'timestamp': True},
            sort={'timestamp': -1},
            limit=5,
        )))
        if not docs:
            return None
        docs.sort(key=lambda d: d.get('timestamp') or d.get('created_at') or '')
        latest = docs[-1]
        return {'answer': latest.get('answer', ''), 'sources': latest.get('sources', [])}

    # --- Pengetahuan & Persona Pengguna ---

    def save_knowledge(self, user_id, question, answer):
        normalized = normalize_question(question)
        doc_id = f'know_{user_id}_{hashlib.sha256(normalized.encode()).hexdigest()[:16]}'
        now_dt = datetime.now(timezone.utc)
        self._upsert_doc(doc_id, {
            'type': 'knowledge',
            'user_id': user_id,
            'question': question,
            'normalized_question': normalized,
            'answer': answer,
            'created_at': now_dt.isoformat(),
            'timestamp': now_dt.timestamp(),
        })

    def find_relevant_knowledge(self, user_id, question, limit=3):
        query_terms = _knowledge_terms(question)
        if len(query_terms) < 2:
            return []
        docs = _call_with_retry(lambda: list(
            self.collection.find({'type': 'knowledge', 'user_id': user_id}, limit=100)
        ))
        matches = []
        for row in docs:
            q_text = row.get('question', '')
            shared = query_terms & _knowledge_terms(q_text)
            coverage = len(shared) / len(query_terms)
            if len(shared) >= 2 and coverage >= 0.5:
                matches.append((coverage, row.get('timestamp') or row.get('created_at') or '', {
                    'question': q_text, 'answer': row.get('answer', ''),
                }))
        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in matches[:limit]]

    def get_persona(self, user_id):
        row = _call_with_retry(lambda: self.collection.find_one({'_id': f'persona_{user_id}'}))
        return row.get('persona', '') if row else ''

    def save_persona(self, user_id, persona):
        if not persona:
            self.delete_persona(user_id)
            return
        self._upsert_doc(f'persona_{user_id}', {
            'type': 'persona',
            'user_id': user_id,
            'persona': persona,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        })

    def delete_persona(self, user_id):
        _call_with_retry(lambda: self.collection.delete_one({'_id': f'persona_{user_id}'}))

    # --- Metadata Ingesti OCR & Auto-Learn RAG ---

    def create_ingestion_job(self, *, user_id, filename, source_title,
                             topic='', scope='user', job_id=None, doc_id=None):
        jid = job_id or str(uuid.uuid4())
        did = doc_id or f'doc_{uuid.uuid4().hex[:16]}'
        now_dt = datetime.now(timezone.utc)
        now_iso = now_dt.isoformat()
        doc = {
            '_id': f'job_{jid}',
            'type': 'ingestion_job',
            'job_id': jid,
            'doc_id': did,
            'user_id': user_id,
            'filename': filename,
            'source_title': source_title,
            'topic': topic,
            'scope': scope,
            'status': 'queued',
            'total_pages': 0,
            'chunks_inserted': 0,
            'error_message': '',
            'created_at': now_iso,
            'updated_at': now_iso,
            'timestamp': now_dt.timestamp(),
        }
        _call_with_retry(lambda: self.collection.insert_one(doc))
        return self._clean_doc(doc)

    def update_ingestion_job(self, job_id, *, status, total_pages=None,
                             chunks_inserted=None, error_message=None):
        fields = {'status': status, 'updated_at': datetime.now(timezone.utc).isoformat()}
        if total_pages is not None:
            fields['total_pages'] = int(total_pages)
        if chunks_inserted is not None:
            fields['chunks_inserted'] = int(chunks_inserted)
        if error_message is not None:
            fields['error_message'] = str(error_message)[:500]
        _call_with_retry(
            lambda: self.collection.update_one({'_id': f'job_{job_id}'}, {'$set': fields})
        )

    def get_ingestion_job(self, job_id):
        row = _call_with_retry(
            lambda: self.collection.find_one({'_id': f'job_{job_id}', 'type': 'ingestion_job'})
        )
        return self._clean_doc(row)

    def get_document_record(self, doc_id):
        row = _call_with_retry(lambda: self.collection.find_one(
            {'type': 'ingestion_job', 'doc_id': doc_id}, sort={'timestamp': -1}
        ))
        if row is not None:
            return self._clean_doc(row)
        auto_row = _call_with_retry(
            lambda: self.collection.find_one({'_id': f'auto_{doc_id}', 'type': 'auto_learned'})
        )
        if auto_row is None:
            return None
        return {
            'doc_id': auto_row['doc_id'],
            'user_id': 'system_auto_kka',
            'scope': 'global',
            'topic': auto_row.get('topic', ''),
            'chunks_inserted': auto_row.get('chunks_inserted', 1),
            'created_at': auto_row.get('created_at', ''),
        }

    def has_auto_learned_key(self, key):
        found = _call_with_retry(lambda: self.collection.find_one(
            {'type': 'auto_learned', '$or': [{'normalized_key': key}, {'question_key': key}]},
            projection={'_id': True},
        ))
        return found is not None

    def save_auto_learned_record(self, *, doc_id, normalized_key, question_key='',
                                 topic, question, summary_markdown,
                                 prior_similarity=None, chunks_inserted=1):
        if self.has_auto_learned_key(normalized_key):
            return False
        now_dt = datetime.now(timezone.utc)
        doc = {
            '_id': f'auto_{doc_id}',
            'type': 'auto_learned',
            'doc_id': doc_id,
            'normalized_key': normalized_key,
            'question_key': question_key,
            'topic': topic,
            'question': question,
            'summary_markdown': summary_markdown,
            'prior_similarity': prior_similarity,
            'chunks_inserted': int(chunks_inserted),
            'created_at': now_dt.isoformat(),
            'timestamp': now_dt.timestamp(),
        }
        try:
            _call_with_retry(lambda: self.collection.insert_one(doc))
            return True
        except Exception as exc:
            if 'duplicate' in str(exc).lower() or 'already exists' in str(exc).lower():
                return False
            raise

    def list_auto_learned_records(self, limit=50):
        safe_limit = max(1, min(int(limit), 200))
        docs = _call_with_retry(lambda: list(self.collection.find(
            {'type': 'auto_learned'},
            projection={
                'doc_id': True, 'topic': True, 'question': True,
                'summary_markdown': True, 'prior_similarity': True,
                'chunks_inserted': True, 'created_at': True, 'timestamp': True,
            },
            sort={'timestamp': -1},
            limit=safe_limit,
        )))
        docs.sort(key=lambda d: d.get('timestamp') or d.get('created_at') or '', reverse=True)
        return [self._clean_doc(d) for d in docs[:safe_limit]]

    def list_user_documents(self, user_id):
        docs = _call_with_retry(lambda: list(self.collection.find(
            {'type': 'ingestion_job', 'user_id': user_id},
            sort={'timestamp': -1},
            limit=100,
        )))
        docs.sort(key=lambda d: d.get('timestamp') or d.get('created_at') or '', reverse=True)
        return [self._clean_doc(d) for d in docs]

    def delete_document_record(self, doc_id, *, user_id=None, is_admin=False):
        if is_admin:
            filt = {'doc_id': doc_id, 'type': {'$in': ['ingestion_job', 'auto_learned']}}
        else:
            filt = {'doc_id': doc_id, 'user_id': user_id, 'type': 'ingestion_job', 'scope': 'user'}
        res = _call_with_retry(lambda: self.collection.delete_many(filt))
        return int(getattr(res, 'deleted_count', 0) or 0)

    def delete_user_data(self, user_id):
        _call_with_retry(lambda: self.collection.delete_many({
            'user_id': user_id,
            '$or': [
                {'type': {'$in': ['session', 'exchange', 'knowledge', 'persona']}},
                {'type': 'ingestion_job', 'scope': 'user'},
            ],
        }))
