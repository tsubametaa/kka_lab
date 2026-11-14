"""RAG context builder, topic-shift guard, and conversational history filtering."""

import re

from app.features.memory.gate import _knowledge_terms, normalize_question

SHORT_QUESTION_WORDS = 6

_STANDALONE_OPENER_RE = re.compile(
    r'^\s*(?:apa(?:\s+(?:itu|arti|artinya|pengertian|bedanya|perbedaan))?'
    r'|apakah|jelaskan|uraikan|sebutkan|bandingkan|ceritakan|pengertian|definisi'
    r'|perbedaan|bedanya|(?:bagaimana|gimana)\s+cara\s+kerja'
    r'|what\s+(?:is|are|does)|explain|define|describe|compare|difference\s+between'
    r'|how\s+(?:does|do)|tell\s+me\s+about)\b',
    flags=re.IGNORECASE,
)

_FOLLOW_UP_MODIFIERS = {
    'contoh', 'contohnya', 'maksud', 'maksudnya', 'arti', 'artinya',
    'beda', 'bedanya', 'perbedaan', 'perbedaannya', 'langkah', 'langkahnya',
    'cara', 'caranya', 'fungsi', 'fungsinya', 'tujuan', 'tujuannya',
    'kelebihan', 'kelebihannya', 'kekurangan', 'kekurangannya',
    'dampak', 'dampaknya', 'solusi', 'solusinya', 'kode', 'kodenya',
    'rumus', 'rumusnya', 'detail', 'detailnya', 'lanjut', 'lanjutan',
    'lanjutkan', 'terus', 'teruskan', 'lalu', 'berikutnya', 'selanjutnya',
    'lagi', 'ulang', 'ulangi', 'ringkas', 'singkat', 'sederhana', 'awam',
    'itu', 'ini', 'tersebut', 'tadi', 'gitu', 'begitu', 'sebelumnya',
    'kenapa', 'mengapa', 'gimana', 'bagaimana', 'kok', 'kalau', 'jika', 'misal',
    'example', 'examples', 'meaning', 'difference', 'steps', 'step',
    'code', 'reason', 'why', 'how', 'more', 'further', 'again',
    'elaborate', 'continue', 'details', 'it', 'this', 'that', 'those',
    'these', 'them', 'they', 'same', 'simpler', 'previous',
}

_ANAPHORIC_CUES = {
    'contohnya', 'maksudnya', 'bedanya', 'perbedaannya', 'langkahnya',
    'caranya', 'fungsinya', 'tujuannya', 'kelebihannya', 'kekurangannya',
    'dampaknya', 'solusinya', 'kodenya', 'rumusnya', 'artinya', 'kerjanya',
    'keduanya', 'tersebut', 'tadi', 'gitu', 'begitu', 'lanjut', 'lanjutan',
    'lanjutkan', 'teruskan', 'ulangi', 'sebelumnya', 'berikutnya',
    'selanjutnya', 'it', 'this', 'that', 'those', 'these', 'them', 'same',
    'previous', 'elaborate', 'continue',
}


def _has_anaphoric_cue(question):
    words = normalize_question(question).split()
    return any(w in _ANAPHORIC_CUES for w in words)


def is_follow_up_question(question, topic=''):
    """Deteksi apakah pertanyaan masih melanjutkan topik sebelumnya atau topik baru."""
    if not topic or not topic.strip() or not question or not question.strip():
        return False

    q_words = normalize_question(question).split()
    if not q_words:
        return False

    q_terms = _knowledge_terms(question)
    t_terms = _knowledge_terms(topic)

    # 1. Ada irisan istilah kunci dengan pertanyaan sebelumnya -> masih satu topik
    if q_terms & t_terms:
        return True

    # 2. Kata ganti/referensi eksplisit tanpa konsep baru (mis. "apa itu?", "jelaskan lebih detail")
    non_generic_terms = q_terms - _FOLLOW_UP_MODIFIERS
    if not non_generic_terms:
        return True

    # 3. Penanda anafora/lanjutan eksplisit ("contohnya di Python?", "how does it work?")
    if _has_anaphoric_cue(question):
        return True

    # 4. Pertanyaan pembuka mandiri dengan istilah topik baru yang tak beririsan -> topik baru
    if _STANDALONE_OPENER_RE.search(question):
        return False

    # 5. Pertanyaan panjang (>= 6 kata) tanpa keterkaitan di atas -> topik mandiri/baru
    if len(q_words) >= SHORT_QUESTION_WORDS:
        return False

    # 6. Pertanyaan pendek (< 6 kata): hanya gabung jika eliptis (<=1 konsep baru + kata hubung/tanya)
    return len(non_generic_terms) <= 1 and q_words[0] in (
        'kalau', 'jika', 'misal', 'kenapa', 'kok', 'lalu', 'terus', 'bagaimana',
        'gimana', 'why', 'how', 'with', 'using', 'for', 'di', 'pakai',
    )


def retrieval_query(question, topic=''):
    """Gabungkan pertanyaan follow-up pendek (<6 kata) dengan topik aktifnya; topik baru dibiarkan bersih."""
    if topic and len(question.split()) < SHORT_QUESTION_WORDS and is_follow_up_question(question, topic):
        return f'{topic} {question}'
    return question


def resolve_query_topic(question, topic=''):
    """Mengembalikan (query_rag, topik_berikutnya) dengan pengaman perpindahan topik."""
    query = retrieval_query(question, topic)
    next_topic = topic if query != question else question
    return query, next_topic


def _turns_related(current_question, prev_user, prev_assistant):
    if is_follow_up_question(current_question, prev_user):
        return True
    if _STANDALONE_OPENER_RE.search(current_question):
        return False
    q_terms = _knowledge_terms(current_question)
    if len(q_terms) >= 1 and len(q_terms & _knowledge_terms(prev_assistant)) >= 2:
        return True
    return False


def filter_relevant_history(question, history, max_turns=4):
    """Ambil hanya rentetan turn history terakhir yang masih satu topik dengan pertanyaan baru."""
    if not history or not question or not question.strip():
        return []

    turns = []
    idx = 0
    while idx < len(history):
        msg = history[idx]
        if (
            msg.get('role') == 'user'
            and idx + 1 < len(history)
            and history[idx + 1].get('role') == 'assistant'
        ):
            turns.append([msg, history[idx + 1]])
            idx += 2
        else:
            turns.append([msg])
            idx += 1

    kept = []
    probe = question
    for turn in reversed(turns[-max(1, int(max_turns)):]):
        turn_user = next((m.get('content', '') for m in turn if m.get('role') == 'user'), '')
        turn_asst = next((m.get('content', '') for m in turn if m.get('role') == 'assistant'), '')
        if not _turns_related(probe, turn_user, turn_asst):
            break
        kept.insert(0, turn)
        probe = turn_user or turn_asst
    return [msg for turn in kept for msg in turn]


def build_context(docs, tokenizer, max_tokens=900):
    blocks, sources = [], []
    used = 0
    for doc in docs:
        meta = doc.get('metadata') or {}
        text = str(doc.get('text', '')).strip()
        if not text:
            continue
        title = str(doc.get('source_title') or meta.get('source_title') or 'Materi KKA')[:140]
        tag = f'S{len(sources) + 1}'
        header = f'[{tag}] {title}\n'
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        remaining = max_tokens - used - len(header_ids) - 4
        if remaining < 24:
            break
        text_ids = tokenizer.encode(text, add_special_tokens=False)[:remaining]
        body = tokenizer.decode(text_ids, skip_special_tokens=True)
        blocks.append(header + body)
        used += len(header_ids) + len(text_ids) + 4
        sources.append({
            'label': tag, 'id': str(doc.get('_id', '')),
            'title': title, 'similarity': doc.get('$similarity'),
            'pages_pdf': doc.get('pages_pdf') or meta.get('pages_pdf'),
        })
    return '\n\n'.join(blocks), sources
