"""Validation logic for LoRA pedagogical training dataset rows."""

import re

from app.features.chat.prompts import POLICIES


def _validate_pedagogical_target(answer, row_id):
    steps = re.findall(r'(?<!\S)(?:\d+[.)]|[-*])\s+\S', answer)
    if len(steps) < 2:
        raise ValueError(f'Jawaban target {row_id} harus memuat minimal dua langkah.')

    normalized = ' '.join(re.findall(r'\w+', answer.casefold(), flags=re.UNICODE))
    reasoning_markers = (
        'karena', 'sebab', 'maka', 'sehingga', 'berdasarkan',
        'because', 'since', 'therefore', 'thus',
    )
    if not any(marker in normalized.split() for marker in reasoning_markers):
        raise ValueError(f'Jawaban target {row_id} harus menjelaskan alasan.')

    thinking_markers = (
        'menurutmu', 'mengapa', 'alasan', 'jelaskan', 'bandingkan',
        'perkirakan', 'prediksi', 'buktikan', 'apa', 'siapa', 'bagaimana',
        'why', 'what', 'how', 'compare', 'predict', 'justify', 'explain',
        'which', 'who', 'when', 'where',
    )
    if '?' not in answer or not any(marker in normalized.split() for marker in thinking_markers):
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
        _validate_pedagogical_target(msgs[-1]['content'], row['id'])
        group_modes.setdefault(row['group_id'], set()).add(row['mode'])
    for group_id, modes in group_modes.items():
        if modes != {'umum', 'terstruktur'}:
            raise ValueError(f'Kelompok {group_id} harus mencakup dua mode.')

