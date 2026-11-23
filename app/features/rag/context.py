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
    'risikonya', 'alasannya', 'jawabannya', 'hasilnya', 'efeknya', 'akibatnya',
    'keduanya', 'tersebut', 'tadi', 'gitu', 'begitu', 'lanjut', 'lanjutan',
    'lanjutkan', 'teruskan', 'ulangi', 'sebelumnya', 'berikutnya',
    'selanjutnya', 'memangnya', 'emangnya', 'emang', 'it', 'this', 'that',
    'those', 'these', 'them', 'same', 'previous', 'elaborate', 'continue',
}

_ASSISTANT_REPLY_PHRASES = (
    'tidak tahu', 'nggak tahu', 'gak tau', 'ga tau', 'ndak tahu',
    'belum tahu', 'kurang tahu', 'kurang paham', 'belum paham',
    'tidak paham', 'nggak paham', 'gak paham', 'bingung',
    'memangnya', 'emangnya', 'jawabannya apa', 'apa jawabannya',
    'maksudnya apa', 'maksudnya gimana', 'maksudnya bagaimana',
    'kenapa begitu', 'kenapa gitu', 'mengapa begitu',
    'langsung jelaskan', 'tolong jelaskan saja', 'kasih tahu saja',
    'benar ga', 'benar gak', 'benar nggak', 'betulkah', 'sudah benar',
    'don t know', 'do not know', 'not sure', 'no idea',
    'what is the answer', 'what are the risks', 'what do you mean', 'why is that',
)


def _stem_term(word):
    for suffix in ('nya', 'kah', 'lah', 'pun', 'ku', 'mu'):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def _stemmed_terms(text):
    return {_stem_term(w) for w in _knowledge_terms(text)}


def _has_anaphoric_cue(question):
    words = normalize_question(question).split()
    return any(w in _ANAPHORIC_CUES for w in words)


def _is_assistant_reply(question, prev_assistant=''):
    """Deteksi apakah siswa menjawab atau bertanya balik terhadap pertanyaan asisten sebelumnya."""
    if not question or not question.strip():
        return False
    normalized = normalize_question(question)
    if not normalized:
        return False
    if _STANDALONE_OPENER_RE.search(question):
        return False
    if any(phrase in normalized for phrase in _ASSISTANT_REPLY_PHRASES):
        return True
    if prev_assistant and '?' in prev_assistant:
        q_stems = _stemmed_terms(question)
        a_stems = _stemmed_terms(prev_assistant)
        if q_stems & a_stems:
            return True
    return False


def _extract_assistant_focus(assistant_message, max_chars=260):
    """Ambil kalimat penutup/pertanyaan utama dari respons asisten sebelumnya."""
    cleaned = ' '.join((assistant_message or '').split())
    if not cleaned:
        return ''
    if len(cleaned) <= max_chars:
        return cleaned
    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    tail_parts = []
    total = 0
    for sent in reversed(sentences):
        if not sent:
            continue
        if tail_parts and total + len(sent) + 1 > max_chars:
            break
        tail_parts.insert(0, sent)
        total += len(sent) + 1
    result = ' '.join(tail_parts).strip()
    if len(result) > max_chars:
        result = result[-max_chars:].lstrip()
    return result


def is_follow_up_question(question, topic=''):
    """Deteksi apakah pertanyaan masih melanjutkan topik sebelumnya atau topik baru."""
    if not topic or not topic.strip() or not question or not question.strip():
        return False

    q_words = normalize_question(question).split()
    if not q_words:
        return False

    q_terms = _stemmed_terms(question)
    t_terms = _stemmed_terms(topic)

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


def retrieval_query(question, topic='', assistant_message=''):
    """Gabungkan pertanyaan follow-up atau balasan terhadap pertanyaan asisten dengan topik aktif."""
    clean_q = (question or '').strip()
    clean_topic = (topic or '').strip()
    clean_asst = (assistant_message or '').strip()

    if clean_asst and (_is_assistant_reply(clean_q, clean_asst) or (clean_topic and is_follow_up_question(clean_q, clean_topic))):
        asst_focus = _extract_assistant_focus(clean_asst)
        parts = []
        if clean_topic:
            parts.append(clean_topic[:140].strip())
        if asst_focus:
            parts.append(asst_focus)
        parts.append(clean_q)
        return ' '.join(parts)

    if clean_topic and len(clean_q.split()) < SHORT_QUESTION_WORDS and is_follow_up_question(clean_q, clean_topic):
        return f'{clean_topic} {clean_q}'
    return clean_q


def resolve_query_topic(question, topic=''):
    """Mengembalikan (query_rag, topik_berikutnya) dengan pengaman perpindahan topik."""
    query = retrieval_query(question, topic)
    next_topic = topic if query != question else question
    return query, next_topic


def _turns_related(current_question, prev_user, prev_assistant):
    if is_follow_up_question(current_question, prev_user):
        return True
    if _is_assistant_reply(current_question, prev_assistant):
        return True
    if _STANDALONE_OPENER_RE.search(current_question):
        return False
    q_terms = _stemmed_terms(current_question)
    a_terms = _stemmed_terms(prev_assistant)
    if len(q_terms) >= 1 and len(q_terms & a_terms) >= 1:
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


_LEADING_META_LINE_RE = re.compile(
    r'^(?:Topik:|Sumber:|Materi:|\[[^\n\]]+>[^\n\]]+\])\s*.*$',
    re.IGNORECASE,
)
_PAGE_MARKER_RE = re.compile(r'[ \t]*\[Halaman\s+buku\s+\d+\][ \t]*', re.IGNORECASE)


def _clean_chunk_text(text):
    """Strip noisy chunk metadata headers and PDF page stamps cleanly."""
    lines = text.splitlines()
    while lines and _LEADING_META_LINE_RE.match(lines[0].strip()):
        lines.pop(0)
    cleaned_lines = []
    for line in lines:
        stripped_line = re.sub(r' {2,}', ' ', _PAGE_MARKER_RE.sub(' ', line)).strip()
        if stripped_line:
            cleaned_lines.append(stripped_line)
        elif cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append('')
    cleaned = '\n'.join(cleaned_lines).strip()
    return cleaned or text.strip()


def _truncate_at_boundary(text, tokenizer, max_tokens):
    """Trim text to fit max_tokens cleanly at a sentence or word boundary."""
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(all_ids) <= max_tokens:
        return text, len(all_ids)
    decoded = tokenizer.decode(all_ids[:max_tokens], skip_special_tokens=True).rstrip()
    boundaries = list(re.finditer(r'[.!?](?=\s|$)|\n', decoded))
    if boundaries and boundaries[-1].end() >= len(decoded) * 0.4:
        trimmed = decoded[:boundaries[-1].end()].strip()
    else:
        space_idx = decoded.rfind(' ')
        trimmed = decoded[:space_idx].rstrip() if space_idx >= len(decoded) * 0.4 else decoded
    final_ids = tokenizer.encode(trimmed, add_special_tokens=False)
    return trimmed, min(len(final_ids), max_tokens)


_DOC_REF_RE = re.compile(
    r'\b(?:pdf|docx?|html?|file|berkas|dokumen|lampiran|unggah(?:an)?|upload(?:ed)?|kirim(?:an)?'
    r'|ringkas(?:an)?|rangkum(?:an)?|summarize|summary|attachment|attached)\b',
    re.IGNORECASE,
)


def is_document_reference_query(question, user_docs=None):
    """Detect whether the student is referring to their uploaded document."""
    text = (question or '').strip()
    if not text:
        return False
    if _DOC_REF_RE.search(text):
        return True
    lower = text.lower()
    for doc in user_docs or ():
        for field in ('source_title', 'filename'):
            val = str(doc.get(field) or '').strip().lower()
            stem = val.rsplit('.', 1)[0] if '.' in val else val
            if len(stem) >= 4 and stem in lower:
                return True
    return False


def build_context(docs, tokenizer, max_tokens=900):
    blocks, sources = [], []
    used = 0
    for doc in docs:
        meta = doc.get('metadata') or {}
        raw_text = str(doc.get('text', '')).strip()
        if not raw_text:
            continue
        text = _clean_chunk_text(raw_text)
        title = str(doc.get('source_title') or meta.get('source_title') or 'Materi KKA')[:140]
        tag = f'S{len(sources) + 1}'
        is_user_doc = meta.get('scope') == 'user'
        header = f'Materi Unggahan Siswa ({title}):\n' if is_user_doc else f'Materi ({title}):\n'
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        remaining = max_tokens - used - len(header_ids) - 4
        if remaining < 24:
            break
        body, body_tokens = _truncate_at_boundary(text, tokenizer, remaining)
        if not body:
            continue
        blocks.append(header + body)
        used += len(header_ids) + body_tokens + 4
        sources.append({
            'label': tag, 'id': str(doc.get('_id', '')),
            'title': title, 'similarity': doc.get('$similarity'),
            'pages_pdf': doc.get('pages_pdf') or meta.get('pages_pdf'),
            'scope': 'user' if is_user_doc else (meta.get('scope') or 'global'),
        })
    return '\n\n'.join(blocks), sources
