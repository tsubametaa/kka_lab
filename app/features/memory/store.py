"""SQLite-backed conversation store and personal knowledge memory."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from app.features.memory.gate import (
    _knowledge_terms,
    normalize_question,
    training_quality,
)


class ChatStore:
    def __init__(self, path='data/chat_memory.sqlite3'):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection as db:
                yield db
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS sessions_by_user
                    ON sessions(user_id, created_at);
                CREATE TABLE IF NOT EXISTS exchanges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    training_status TEXT NOT NULL,
                    training_reason TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS exchanges_by_user_question
                    ON exchanges(user_id, normalized_question, id DESC);
                CREATE INDEX IF NOT EXISTS exchanges_by_session
                    ON exchanges(session_id, id);
                CREATE TABLE IF NOT EXISTS user_knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    normalized_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, normalized_question)
                );
                CREATE INDEX IF NOT EXISTS knowledge_by_user
                    ON user_knowledge(user_id, id DESC);
                CREATE TABLE IF NOT EXISTS user_personas (
                    user_id TEXT PRIMARY KEY,
                    persona TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_title TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_pages INTEGER NOT NULL DEFAULT 0,
                    chunks_inserted INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS ingestion_by_user
                    ON ingestion_jobs(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS ingestion_by_doc
                    ON ingestion_jobs(doc_id);
                CREATE TABLE IF NOT EXISTS auto_learned_rag (
                    doc_id TEXT PRIMARY KEY,
                    normalized_key TEXT NOT NULL UNIQUE,
                    question_key TEXT NOT NULL DEFAULT '',
                    topic TEXT NOT NULL,
                    question TEXT NOT NULL,
                    summary_markdown TEXT NOT NULL,
                    prior_similarity REAL,
                    chunks_inserted INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS auto_learned_by_qkey
                    ON auto_learned_rag(question_key);
            ''')

    def create_session(self, user_id):
        session_id = str(uuid.uuid4())
        with self._connection() as db:
            db.execute('INSERT INTO sessions (id, user_id) VALUES (?, ?)',
                       (session_id, user_id))
        return session_id

    def session_belongs_to(self, session_id, user_id):
        with self._connection() as db:
            row = db.execute(
                'SELECT 1 FROM sessions WHERE id = ? AND user_id = ?',
                (session_id, user_id),
            ).fetchone()
        return row is not None

    def history(self, session_id, limit=4):
        with self._connection() as db:
            rows = db.execute(
                '''SELECT question, answer FROM exchanges
                   WHERE session_id = ? ORDER BY id DESC LIMIT ?''',
                (session_id, limit),
            ).fetchall()
        history = []
        for row in reversed(rows):
            history.extend((
                {'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': row['answer']},
            ))
        return history

    def find_repeated_question(self, user_id, question, mode=None):
        normalized = normalize_question(question)
        if len(normalized.split()) < 3:
            return None
        with self._connection() as db:
            row = db.execute(
                '''SELECT answer, sources_json FROM exchanges
                   WHERE user_id = ? AND normalized_question = ?
                     AND (? IS NULL OR mode = ?)
                   ORDER BY id DESC LIMIT 1''',
                (user_id, normalized, mode, mode),
            ).fetchone()
        if row is None:
            return None
        return {'answer': row['answer'], 'sources': json.loads(row['sources_json'])}

    def save_exchange(self, user_id, session_id, question, answer, mode,
                      sources, training_consent, memory_replay=False):
        eligible, reason = training_quality(question, answer, training_consent)
        if memory_replay:
            eligible, reason = False, 'answer_reused_from_memory'
        status = 'pending_review' if eligible else 'excluded'
        with self._connection() as db:
            db.execute(
                '''INSERT INTO exchanges
                   (user_id, session_id, question, normalized_question, answer,
                    mode, sources_json, training_status, training_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (user_id, session_id, question, normalize_question(question),
                 answer, mode, json.dumps(sources, ensure_ascii=False), status, reason),
            )
        return {'eligible_for_review': eligible, 'status': status, 'reason': reason}

    def save_knowledge(self, user_id, question, answer):
        normalized = normalize_question(question)
        with self._connection() as db:
            db.execute(
                '''INSERT INTO user_knowledge
                   (user_id, question, normalized_question, answer)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, normalized_question) DO UPDATE SET
                       question = excluded.question,
                       answer = excluded.answer,
                       created_at = CURRENT_TIMESTAMP''',
                (user_id, question, normalized, answer),
            )

    def find_relevant_knowledge(self, user_id, question, limit=3):
        query_terms = _knowledge_terms(question)
        if len(query_terms) < 2:
            return []
        with self._connection() as db:
            rows = db.execute(
                '''SELECT question, answer, id FROM user_knowledge
                   WHERE user_id = ? ORDER BY id DESC''',
                (user_id,),
            ).fetchall()

        matches = []
        for row in rows:
            shared = query_terms & _knowledge_terms(row['question'])
            coverage = len(shared) / len(query_terms)
            if len(shared) >= 2 and coverage >= 0.5:
                matches.append((coverage, row['id'], {
                    'question': row['question'], 'answer': row['answer'],
                }))
        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in matches[:limit]]

    def get_persona(self, user_id):
        with self._connection() as db:
            row = db.execute(
                'SELECT persona FROM user_personas WHERE user_id = ?', (user_id,)
            ).fetchone()
        return row['persona'] if row else ''

    def save_persona(self, user_id, persona):
        """Simpan persona yang SUDAH divalidasi; teks kosong menghapus persona."""
        if not persona:
            self.delete_persona(user_id)
            return
        with self._connection() as db:
            db.execute(
                '''INSERT INTO user_personas (user_id, persona) VALUES (?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       persona = excluded.persona,
                       updated_at = CURRENT_TIMESTAMP''',
                (user_id, persona),
            )

    def delete_persona(self, user_id):
        with self._connection() as db:
            db.execute('DELETE FROM user_personas WHERE user_id = ?', (user_id,))

    def create_ingestion_job(self, *, user_id, filename, source_title,
                             topic='', scope='user', job_id=None, doc_id=None):
        jid = job_id or str(uuid.uuid4())
        did = doc_id or f'doc_{uuid.uuid4().hex[:16]}'
        with self._connection() as db:
            db.execute(
                '''INSERT INTO ingestion_jobs
                   (job_id, doc_id, user_id, filename, source_title, topic, scope, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'queued')''',
                (jid, did, user_id, filename, source_title, topic, scope),
            )
        return self.get_ingestion_job(jid)

    def update_ingestion_job(self, job_id, *, status, total_pages=None,
                             chunks_inserted=None, error_message=None):
        fields = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
        values = [status]
        if total_pages is not None:
            fields.append('total_pages = ?')
            values.append(int(total_pages))
        if chunks_inserted is not None:
            fields.append('chunks_inserted = ?')
            values.append(int(chunks_inserted))
        if error_message is not None:
            fields.append('error_message = ?')
            values.append(str(error_message)[:500])
        values.append(job_id)
        with self._connection() as db:
            db.execute(
                f"UPDATE ingestion_jobs SET {', '.join(fields)} WHERE job_id = ?",
                values,
            )

    def get_ingestion_job(self, job_id):
        with self._connection() as db:
            row = db.execute(
                'SELECT * FROM ingestion_jobs WHERE job_id = ?', (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_document_record(self, doc_id):
        with self._connection() as db:
            row = db.execute(
                'SELECT * FROM ingestion_jobs WHERE doc_id = ? ORDER BY created_at DESC LIMIT 1',
                (doc_id,),
            ).fetchone()
            if row is not None:
                return dict(row)
            auto_row = db.execute(
                'SELECT doc_id, topic, chunks_inserted, created_at FROM auto_learned_rag WHERE doc_id = ?',
                (doc_id,),
            ).fetchone()
        if auto_row is None:
            return None
        return {
            'doc_id': auto_row['doc_id'],
            'user_id': 'system_auto_kka',
            'scope': 'global',
            'topic': auto_row['topic'],
            'chunks_inserted': auto_row['chunks_inserted'],
            'created_at': auto_row['created_at'],
        }

    def has_auto_learned_key(self, key):
        with self._connection() as db:
            row = db.execute(
                'SELECT 1 FROM auto_learned_rag WHERE normalized_key = ? OR question_key = ?',
                (key, key),
            ).fetchone()
        return row is not None

    def save_auto_learned_record(self, *, doc_id, normalized_key, question_key='',
                                 topic, question, summary_markdown,
                                 prior_similarity=None, chunks_inserted=1):
        with self._connection() as db:
            cur = db.execute(
                '''INSERT OR IGNORE INTO auto_learned_rag
                   (doc_id, normalized_key, question_key, topic, question,
                    summary_markdown, prior_similarity, chunks_inserted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (doc_id, normalized_key, question_key, topic, question,
                 summary_markdown, prior_similarity, int(chunks_inserted)),
            )
        return cur.rowcount > 0

    def list_auto_learned_records(self, limit=50):
        with self._connection() as db:
            rows = db.execute(
                '''SELECT doc_id, topic, question, summary_markdown,
                          prior_similarity, chunks_inserted, created_at
                   FROM auto_learned_rag
                   ORDER BY created_at DESC LIMIT ?''',
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_user_documents(self, user_id):
        with self._connection() as db:
            rows = db.execute(
                '''SELECT job_id, doc_id, filename, source_title, topic, scope,
                          status, total_pages, chunks_inserted, error_message,
                          created_at, updated_at
                   FROM ingestion_jobs
                   WHERE user_id = ? ORDER BY created_at DESC''',
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document_record(self, doc_id, *, user_id=None, is_admin=False):
        with self._connection() as db:
            if is_admin:
                cur1 = db.execute('DELETE FROM ingestion_jobs WHERE doc_id = ?', (doc_id,))
                cur2 = db.execute('DELETE FROM auto_learned_rag WHERE doc_id = ?', (doc_id,))
                return cur1.rowcount + cur2.rowcount
            cur = db.execute(
                "DELETE FROM ingestion_jobs WHERE doc_id = ? AND user_id = ? AND scope = 'user'",
                (doc_id, user_id),
            )
        return cur.rowcount

    def delete_user_data(self, user_id):
        with self._connection() as db:
            db.execute('PRAGMA foreign_keys = ON')
            db.execute('DELETE FROM sessions WHERE user_id = ?', (user_id,))
            db.execute('DELETE FROM user_knowledge WHERE user_id = ?', (user_id,))
            db.execute('DELETE FROM user_personas WHERE user_id = ?', (user_id,))
            db.execute(
                "DELETE FROM ingestion_jobs WHERE user_id = ? AND scope = 'user'",
                (user_id,),
            )

