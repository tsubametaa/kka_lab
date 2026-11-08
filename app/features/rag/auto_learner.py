"""Guarded auto-learning pipeline: filters KKA discussions and distills textbook notes into RAG."""

import hashlib
import json
import logging
import os
import re

from app.features.language.translator import _generate_local_text
from app.features.memory.gate import normalize_question, training_quality
from app.features.ocr.client import PageMarkdown
from app.features.rag.chunker import chunk_markdown_pages
from app.features.rag.ingestor import ingest_chunks

logger = logging.getLogger(__name__)

AUTO_LEARN_SOURCE_TITLE = 'Catatan Materi KKA (Diskusi Terkurasi)'
DEFAULT_MAX_PRIOR_SIMILARITY = 0.82
MIN_SUBSTANTIVE_ANSWER_CHARS = 120

# Kosakata inti kurikulum Koding dan Kecerdasan Artifisial (KKA) Indonesia & English.
KKA_SINGLE_TERMS = frozenset({
    # Berpikir komputasional & algoritma
    'kka', 'algoritma', 'algorithm', 'flowchart', 'pseudocode', 'dekomposisi',
    'decomposition', 'abstraksi', 'abstraction', 'rekursi', 'recursion',
    'iterasi', 'iteration', 'sorting', 'searching', 'komputasional', 'computational',
    'logika', 'boolean', 'biner', 'binary',
    # Koding & pemrograman
    'koding', 'coding', 'pemrograman', 'programming', 'kode', 'program',
    'python', 'javascript', 'variabel', 'variable', 'konstanta', 'integer',
    'float', 'string', 'array', 'list', 'tuple', 'dictionary', 'fungsi',
    'function', 'parameter', 'argumen', 'perulangan', 'loop', 'for', 'while',
    'percabangan', 'kondisional', 'conditional', 'if', 'elif', 'else',
    'sintaks', 'syntax', 'debugging', 'debug', 'kompilasi', 'compiler',
    'interpreter', 'oop', 'class', 'objek', 'object', 'method', 'modul',
    'library', 'api', 'json', 'sql', 'database',
    # Literasi data
    'dataset', 'validasi', 'validation', 'fitur', 'feature', 'features',
    'label', 'normalisasi', 'normalization', 'outlier', 'preprocessing',
    'imputasi', 'encoding', 'csv', 'statistik',
    # Kecerdasan artifisial & machine learning
    'ai', 'ml', 'dl', 'nlp', 'llm', 'rag', 'ocr', 'knn', 'svm', 'cnn', 'rnn',
    'supervised', 'unsupervised', 'reinforcement', 'klasifikasi', 'classification',
    'regresi', 'regression', 'clustering', 'klasterisasi', 'kmeans',
    'overfitting', 'underfitting', 'akurasi', 'accuracy', 'presisi', 'precision',
    'recall', 'f1', 'loss', 'epoch', 'batch', 'gradient', 'hyperparameter',
    'inferensi', 'inference', 'prediksi', 'prediction', 'model', 'training',
    'token', 'tokenizer', 'embedding', 'vektor', 'vector', 'similarity',
    'prompt', 'prompting', 'halusinasi', 'hallucination', 'lora', 'qlora',
    'chatbot', 'bias', 'datasetnya',
})

KKA_PHRASE_TERMS = (
    'kecerdasan artifisial', 'kecerdasan buatan', 'artificial intelligence',
    'machine learning', 'pembelajaran mesin', 'deep learning',
    'berpikir komputasional', 'computational thinking', 'pengenalan pola',
    'pattern recognition', 'struktur data', 'data structure', 'tipe data',
    'data type', 'data latih', 'data uji', 'training data', 'test data',
    'confusion matrix', 'decision tree', 'pohon keputusan', 'random forest',
    'neural network', 'jaringan saraf', 'k means', 'linear regression',
    'logistic regression', 'computer vision', 'visi komputer', 'fine tuning',
    'etika ai', 'privasi data', 'keamanan data',
)

_POISONING_PATTERNS = (
    r'abaikan\s+(?:semua\s+)?aturan',
    r'lupakan\s+(?:semua\s+)?aturan',
    r'instruksi\s+sebelumnya',
    r'system\s+prompt',
    r'prompt\s+sistem',
    r'ignore\s+(?:all\s+)?(?:previous\s+)?instructions',
    r'jailbreak',
    r'catat\s+(?:ya\s+)?bahwa',
    r'ingat\s+mulai\s+sekarang',
    r'simpan\s+ke\s+(?:database|rag|memori)',
    r'masukkan\s+ke\s+(?:database|rag)',
    r'anggap\s+saja\s+(?:bahwa|rumus|materi)',
    r'aturan\s+baru\s*:',
    r'save\s+(?:this\s+)?to\s+(?:rag|database)',
    r'remember\s+from\s+now\s+on',
    r'\b(?:goblok|tolol|anjing|bangsat|bodoh|mesum|seksual|nsfw)\b',
)

_EXTRA_UNCERTAINTY_MARKERS = (
    'tidak ditemukan di materi',
    'tidak ada di materi',
    'tidak tercantum di sumber',
    'bertanya kepada guru',
    'tanyakan kepada guru',
    'konfirmasi kepada guru',
    'not found in the material',
    'not in the provided',
    'ask your teacher',
    'confirm with your teacher',
)

_JUDGE_SYSTEM_PROMPT = (
    'Anda kurator buku ajar Koding dan Kecerdasan Artifisial (KKA) SMA Kelas X. '
    'Tugas Anda menilai apakah pasangan tanya-jawab berikut layak dijadikan catatan '
    'referensi resmi KKA.\n'
    'TOLAK (approved: false) jika salah satu terjadi:\n'
    '1. Topik di luar kurikulum KKA (mis. obrolan umum, curhat, game, sejarah, biologi, dll).\n'
    '2. Penjelasan mengandung tebakan, keraguan, kesalahan konsep teknis, atau klaim palsu.\n'
    '3. Jawaban hanya berupa pertanyaan balik/sapaan tanpa memaparkan konsep teknis yang utuh.\n\n'
    'SETUJUI (approved: true) HANYA jika penjelasan konsep KKA benar, jelas, dan utuh.\n'
    'Jika disetujui, tulis ulang intisarinya di field "summary_markdown" sebagai catatan '
    'materi buku ajar yang NETRAL dan FAKTUAL (2-5 kalimat atau poin ringkas, diawali '
    'heading "## <Judul Sub-Topik>"). Hapus semua gaya persona, sapaan, label [S1]/[M1], '
    'dan hapus pertanyaan balik ke siswa.\n'
    'Kembalikan HANYA JSON valid dengan struktur persis:\n'
    '{"approved": bool, "topic": "string", "summary_markdown": "string"}'
)


def is_auto_learn_enabled():
    val = os.getenv('RAG_AUTO_LEARN_ENABLED', 'true').strip().lower()
    return val in ('1', 'true', 'yes', 'on')


def max_source_similarity(sources):
    scores = [
        float(item['similarity'])
        for item in (sources or [])
        if isinstance(item, dict) and isinstance(item.get('similarity'), (int, float))
    ]
    return max(scores) if scores else None


def _has_poisoning_or_injection(text):
    lowered = (text or '').casefold()
    return any(re.search(pattern, lowered) for pattern in _POISONING_PATTERNS)


def match_kka_domain_terms(text):
    normalized = normalize_question(text or '')
    if not normalized:
        return set()
    words = set(normalized.split())
    matched = words & KKA_SINGLE_TERMS
    for phrase in KKA_PHRASE_TERMS:
        if f' {phrase} ' in f' {normalized} ':
            matched.add(phrase)
    return matched


def evaluate_kka_gate(question, answer, *, sources=None, memory_hit=False,
                      max_similarity_threshold=None):
    """Saringan deterministik 4 lapis (cepat, tanpa GPU) sebelum masuk verifikasi LLM."""
    if memory_hit:
        return False, 'reused_from_memory'

    q_text = (question or '').strip()
    a_text = (answer or '').strip()
    if _has_poisoning_or_injection(q_text) or _has_poisoning_or_injection(a_text):
        return False, 'blocked_poisoning_or_injection'

    if len(a_text) < MIN_SUBSTANTIVE_ANSWER_CHARS:
        return False, 'answer_not_substantive'

    # Lapis kualitas & kepastian jawaban (tanpa mewajibkan flag consent dataset LoRA).
    eligible, reason = training_quality(q_text, a_text, True)
    if not eligible:
        return False, reason

    lowered_answer = normalize_question(a_text)
    if any(marker in lowered_answer for marker in _EXTRA_UNCERTAINTY_MARKERS):
        return False, 'assistant_uncertain_or_ungrounded'

    # Lapis domain KKA: pertanyaan wajib mengandung minimal 1 istilah KKA
    q_terms = match_kka_domain_terms(q_text)
    if not q_terms:
        return False, 'outside_kka_domain'
    combined_terms = q_terms | match_kka_domain_terms(a_text)
    if len(combined_terms) < 2:
        return False, 'insufficient_kka_depth'

    # Lapis anti-duplikat: jika sudah tercakup kuat di Astra DB, lewati.
    threshold = (
        float(max_similarity_threshold)
        if max_similarity_threshold is not None
        else float(os.getenv('RAG_AUTO_LEARN_MAX_SIMILARITY', str(DEFAULT_MAX_PRIOR_SIMILARITY)))
    )
    best_sim = max_source_similarity(sources)
    if best_sim is not None and best_sim >= threshold:
        return False, 'already_covered_in_rag'

    return True, 'passed_fast_kka_gate'


def parse_and_validate_distilled_note(raw_output):
    if not raw_output:
        return None
    cleaned = re.sub(
        r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw_output.strip(), flags=re.IGNORECASE
    )
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get('approved') is not True:
        return None

    topic = str(data.get('topic') or '').strip()[:120]
    summary = str(data.get('summary_markdown') or '').strip()
    # Bersihkan sisa tag referensi [S1]/[M1] jika masih terbawa.
    summary = re.sub(r'\[(?:S|M)\d+\]', '', summary).strip()

    if not topic or len(summary) < 80 or len(summary) > 1800:
        return None
    if _has_poisoning_or_injection(summary):
        return None
    lowered_summary = normalize_question(summary)
    if any(marker in lowered_summary for marker in _EXTRA_UNCERTAINTY_MARKERS):
        return None
    if len(match_kka_domain_terms(f'{topic} {summary}')) < 2:
        return None
    if not summary.startswith('#'):
        summary = f'## {topic}\n\n{summary}'
    return {'topic': topic, 'summary_markdown': summary}


def verify_and_distill_kka_note(question, answer, model, tokenizer, gpu_lock=None):
    """Gunakan Qwen lokal (greedy decoding) untuk memverifikasi akurasi & menulis ulang catatan netral."""
    if model is None or tokenizer is None:
        return None
    messages = [
        {'role': 'system', 'content': _JUDGE_SYSTEM_PROMPT},
        {
            'role': 'user',
            'content': f'Pertanyaan siswa:\n{question.strip()}\n\nPenjelasan asisten:\n{answer.strip()}',
        },
    ]
    if gpu_lock is not None:
        with gpu_lock:
            raw = _generate_local_text(
                messages, model, tokenizer, max_input_tokens=2048, max_new_tokens=380
            )
    else:
        raw = _generate_local_text(
            messages, model, tokenizer, max_input_tokens=2048, max_new_tokens=380
        )
    return parse_and_validate_distilled_note(raw)


def maybe_auto_learn_from_exchange(store, *, question, answer, sources=None,
                                   memory_hit=False, assistant=None):
    """Background task Pilihan B: saring diskusi KKA baru, verifikasi & netralisasi, lalu simpan ke RAG."""
    try:
        if not is_auto_learn_enabled():
            return {'learned': False, 'reason': 'disabled'}

        passed, reason = evaluate_kka_gate(
            question, answer, sources=sources, memory_hit=memory_hit
        )
        if not passed:
            return {'learned': False, 'reason': reason}

        q_key = normalize_question(question)[:240]
        if store.has_auto_learned_key(q_key):
            return {'learned': False, 'reason': 'duplicate_question'}

        model = getattr(assistant, 'model', None)
        tokenizer = getattr(assistant, 'tokenizer', None)
        gpu_lock = getattr(assistant, 'gpu_lock', None)

        distilled = verify_and_distill_kka_note(
            question, answer, model, tokenizer, gpu_lock=gpu_lock
        )
        if distilled is None:
            return {'learned': False, 'reason': 'rejected_by_kka_judge'}

        topic = distilled['topic']
        summary_md = distilled['summary_markdown']
        topic_key = f"topic:{normalize_question(topic)[:200]}"
        if store.has_auto_learned_key(topic_key):
            return {'learned': False, 'reason': 'duplicate_topic'}

        digest = hashlib.sha256(topic_key.encode('utf-8')).hexdigest()[:14]
        doc_id = f'auto_kka_{digest}'

        chunks = chunk_markdown_pages(
            [PageMarkdown(page_number=1, markdown=summary_md)],
            tokenizer=tokenizer,
            default_topic=topic,
        )
        if not chunks:
            return {'learned': False, 'reason': 'empty_chunks'}

        inserted = ingest_chunks(
            chunks,
            doc_id=doc_id,
            source_title=AUTO_LEARN_SOURCE_TITLE,
            user_id='system_auto_kka',
            scope='global',
            default_topic=topic,
            filename='',
            extra_metadata={
                'origin': 'auto_discussion',
                'verified_by': 'ai_kka_gate',
            },
        )
        prior_sim = max_source_similarity(sources)
        store.save_auto_learned_record(
            doc_id=doc_id,
            normalized_key=topic_key,
            question_key=q_key,
            topic=topic,
            question=question.strip()[:500],
            summary_markdown=summary_md,
            prior_similarity=prior_sim,
            chunks_inserted=inserted,
        )
        logger.info('Auto-learned materi KKA baru ke RAG: %s (%s)', doc_id, topic)
        return {'learned': True, 'doc_id': doc_id, 'topic': topic, 'chunks': inserted}
    except Exception:
        logger.exception('Gagal menjalankan auto-learn RAG dari diskusi')
        return {'learned': False, 'reason': 'internal_error'}
