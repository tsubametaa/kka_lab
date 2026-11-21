"""Validation logic for LoRA pedagogical training dataset rows."""

import re

from app.features.chat.prompts import POLICIES


_STIFF_OPENERS_RE = re.compile(
    r'^\s*(?:pahami(?:\s+bahwa)?|pelajari|kuasai|ketahui|simak|catat)\b'
    r'|^\s*1[.)]\s+(?:identifikasi|tinjau|periksa|analisis|amati|urutkan|catat)\b',
    flags=re.IGNORECASE,
)
_ARTIFACT_LEAK_RE = re.compile(
    r'(?<!\w)\[(?:S|M)\d+(?:\s*[,;]\s*(?:S|M)\d+)*\]'
    r'|<\|[^|]+\|>|<｜[^｜]+｜>'
    r'|^\s*(?:Topik|Konteks Sumber|Memori Personal):'
    r'|\bauto_kka_[0-9a-f]{6,}\b',
    flags=re.IGNORECASE | re.MULTILINE,
)
_UNSTRUCTURED_JUMP_QUESTION_RE = re.compile(
    r'(?:apakah\s+kamu\s+tahu|tahukah\s+kamu|kamu\s+tahu\s+(?:tidak|nggak|gak))\s+'
    r'apa\s+itu\s+[^?.!]+\s+dan\s+(?:bagaimana|gimana)\s+cara\s+kerja\s*(?:nya)?\s*\?'
    r'|\bdo\s+you\s+know\s+what\s+[^?.!]+\s+is\s+and\s+how\s+it\s+works\s*\?',
    flags=re.IGNORECASE,
)

_REASONING_MARKERS = (
    'karena', 'sebab', 'soalnya', 'maka', 'sehingga', 'supaya', 'agar',
    'artinya', 'jadi', 'akibatnya', 'berdasarkan',
    'because', 'since', 'therefore', 'thus', 'so',
)
_EXAMPLE_MARKERS = (
    'misalnya', 'misalkan', 'contohnya', 'contoh', 'bayangkan',
    'ibarat', 'ibaratnya', 'umpama', 'seperti', 'gambaran', 'ilustrasi',
    'simulasi', 'kasus',
    'example', 'imagine', 'suppose', 'such', 'analogy', 'simulation', 'scenario',
)
_THINKING_MARKERS = (
    'menurutmu', 'mengapa', 'kenapa', 'alasan', 'jelaskan', 'bandingkan',
    'perkirakan', 'prediksi', 'buktikan', 'apa', 'apakah', 'siapa',
    'bagaimana', 'mana', 'manakah', 'coba', 'kira', 'kalau', 'jika',
    'ciri', 'langkah', 'perhatikan', 'lihat', 'bedanya',
    'why', 'what', 'how', 'compare', 'predict', 'justify', 'explain',
    'which', 'who', 'when', 'where', 'suppose', 'notice', 'feature', 'step',
)
_LIMITATION_MARKERS = (
    'tidak ditemukan', 'tidak ada di materi', 'tidak tercantum', 'di luar materi',
    'belum dibahas', 'tidak tersedia', 'tanyakan kepada guru', 'konfirmasi kepada guru',
    'konsultasi dengan guru', 'diskusikan dengan guru',
    'not found', 'outside', 'not covered', 'ask your teacher', 'confirm with',
)


def _check_clean_response(text, row_id):
    if _ARTIFACT_LEAK_RE.search(text):
        raise ValueError(
            f'Jawaban target {row_id} membocorkan artefak teknis/referensi.'
        )
    if _STIFF_OPENERS_RE.search(text):
        raise ValueError(
            f'Jawaban target {row_id} memakai pembuka perintah kaku; gunakan bahasa natural.'
        )
    if not (text.endswith('```') or text[-1] in '.!?)"\''):
        raise ValueError(
            f'Jawaban target {row_id} harus diakhiri tanda baca kalimat yang utuh.'
        )


def _validate_pedagogical_target(answer, row_id, mode='umum', sample_type='lesson',
                                 require_question=None, require_example=None):
    stripped = answer.strip()
    _check_clean_response(stripped, row_id)

    lower = stripped.casefold()
    normalized = ' '.join(re.findall(r'\w+', lower, flags=re.UNICODE))
    words = set(normalized.split())
    word_count = len(stripped.split())

    if sample_type == 'concise':
        if word_count < 8:
            raise ValueError(f'Jawaban concise {row_id} terlalu pendek (< 8 kata).')
        if require_question is True and '?' not in stripped:
            raise ValueError(f'Jawaban target {row_id} harus memuat pemantik berpikir.')
        return

    if sample_type == 'refusal':
        if word_count < 10 or not any(m in lower for m in _LIMITATION_MARKERS):
            raise ValueError(
                f'Jawaban refusal {row_id} harus menyatakan keterbatasan materi atau rujukan guru.'
            )
        return

    sentences = [s.strip() for s in re.split(r'[.!?]+', stripped) if len(s.strip().split()) >= 3]
    if len(sentences) < 2 or word_count < 15:
        raise ValueError(
            f'Jawaban target {row_id} harus memuat minimal dua kalimat penjelasan yang memadai.'
        )

    if sample_type == 'multi_point' and mode == 'umum' and word_count < 55:
        raise ValueError(f'Jawaban multi_point {row_id} (mode umum) harus tuntas minimal 55 kata.')

    if not any(marker in words for marker in _REASONING_MARKERS):
        raise ValueError(f'Jawaban target {row_id} harus menjelaskan alasan.')

    need_example = (mode == 'umum' and sample_type == 'lesson') if require_example is None else require_example
    if need_example:
        has_code_block = '```' in stripped or '`' in stripped
        if not has_code_block and not any(marker in words for marker in _EXAMPLE_MARKERS):
            raise ValueError(
                f'Jawaban target {row_id} (mode umum) harus memuat contoh konkret atau analogi.'
            )

    if _UNSTRUCTURED_JUMP_QUESTION_RE.search(stripped):
        raise ValueError(
            f'Pertanyaan pedagogis {row_id} terlalu loncat/sulit sekaligus; '
            'pancing bertahap dari pertanyaan dasar yang mudah dulu.'
        )
    need_question = (sample_type == 'lesson') if require_question is None else require_question
    if need_question:
        if '?' not in stripped or not any(marker in words for marker in _THINKING_MARKERS):
            raise ValueError(f'Jawaban target {row_id} harus memuat pemantik berpikir.')


def validate_rows(rows):
    seen = set()
    group_modes = {}
    for row in rows:
        for key in ('id', 'group_id', 'mode', 'context', 'messages'):
            if key not in row:
                raise ValueError(f'Kolom {key} tidak ditemukan.')
        if row['id'] in seen:
            raise ValueError(f'ID ganda: {row["id"]}')
        seen.add(row['id'])
        if row['mode'] not in POLICIES:
            raise ValueError(f'Mode tidak valid pada {row["id"]}')
        msgs = row['messages']
        if len(msgs) < 2 or len(msgs) % 2:
            raise ValueError('Urutan harus user/assistant dan berakhir assistant.')
        for i, msg in enumerate(msgs):
            expected = 'user' if i % 2 == 0 else 'assistant'
            content = msg.get('content', '').strip()
            if msg.get('role') != expected or not content:
                raise ValueError(f'Pesan tidak valid pada {row["id"]}')
            if expected == 'assistant':
                _check_clean_response(content, row['id'])
        _validate_pedagogical_target(
            msgs[-1]['content'],
            row['id'],
            mode=row['mode'],
            sample_type=row.get('sample_type', 'lesson'),
            require_question=row.get('require_question'),
            require_example=row.get('require_example'),
        )
        group_modes.setdefault(row['group_id'], set()).add(row['mode'])
    for group_id, modes in group_modes.items():
        if modes != {'umum', 'terstruktur'}:
            raise ValueError(f'Kelompok {group_id} harus mencakup dua mode.')