"""Validation logic for LoRA pedagogical training dataset rows."""

import re

from app.features.chat.prompts import POLICIES


_STIFF_OPENERS_RE = re.compile(
    r'^\s*(?:pahami(?:\s+bahwa)?|pelajari|kuasai|ketahui|simak|catat)\b'
    r'|^\s*1[.)]\s+(?:identifikasi|tinjau|periksa|analisis|amati|urutkan|catat)\b',
    flags=re.IGNORECASE,
)

def _validate_pedagogical_target(answer, row_id, mode='umum'):
    stripped = answer.strip()
    if _STIFF_OPENERS_RE.search(stripped):
        raise ValueError(
            f'Jawaban target {row_id} memakai pembuka perintah kaku; gunakan bahasa natural.'
        )

    sentences = [s.strip() for s in re.split(r'[.!?]+', stripped) if len(s.strip().split()) >= 3]
    if len(sentences) < 2 or len(stripped.split()) < 15:
        raise ValueError(
            f'Jawaban target {row_id} harus memuat minimal dua kalimat penjelasan yang memadai.'
        )

    normalized = ' '.join(re.findall(r'\w+', stripped.casefold(), flags=re.UNICODE))
    words = set(normalized.split())
    reasoning_markers = (
        'karena', 'sebab', 'soalnya', 'maka', 'sehingga', 'supaya', 'agar',
        'artinya', 'jadi', 'akibatnya', 'berdasarkan',
        'because', 'since', 'therefore', 'thus', 'so',
    )
    if not any(marker in words for marker in reasoning_markers):
        raise ValueError(f'Jawaban target {row_id} harus menjelaskan alasan.')

    if mode == 'umum':
        example_markers = (
            'misalnya', 'misalkan', 'contohnya', 'contoh', 'bayangkan',
            'ibarat', 'ibaratnya', 'umpama', 'seperti', 'gambaran', 'ilustrasi',
            'simulasi', 'kasus',
            'example', 'imagine', 'suppose', 'such', 'analogy', 'simulation', 'scenario',
        )
        has_code_block = '```' in stripped or '`' in stripped
        if not has_code_block and not any(marker in words for marker in example_markers):
            raise ValueError(
                f'Jawaban target {row_id} (mode umum) harus memuat contoh konkret atau analogi.'
            )

    thinking_markers = (
        'menurutmu', 'mengapa', 'kenapa', 'alasan', 'jelaskan', 'bandingkan',
        'perkirakan', 'prediksi', 'buktikan', 'apa', 'apakah', 'siapa',
        'bagaimana', 'mana', 'manakah', 'coba', 'kira', 'kalau', 'jika',
        'why', 'what', 'how', 'compare', 'predict', 'justify', 'explain',
        'which', 'who', 'when', 'where', 'suppose',
    )
    if '?' not in stripped or not any(marker in words for marker in thinking_markers):
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
            if msg.get('role') != expected or not msg.get('content', '').strip():
                raise ValueError(f'Pesan tidak valid pada {row["id"]}')
        _validate_pedagogical_target(msgs[-1]['content'], row['id'], mode=row['mode'])
        group_modes.setdefault(row['group_id'], set()).add(row['mode'])
    for group_id, modes in group_modes.items():
        if modes != {'umum', 'terstruktur'}:
            raise ValueError(f'Kelompok {group_id} harus mencakup dua mode.')