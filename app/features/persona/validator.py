"""Validation for student-defined answer personas (tone/style only).

Persona hanya mengatur gaya bahasa. Aturan pedagogis, sumber, bahasa, dan
keamanan di prompt sistem tetap berlaku. Validasi ini heuristik (daftar pola),
bukan moderasi lengkap; guru tetap perlu memantau penggunaan.
"""

import re
import unicodedata

MAX_PERSONA_CHARS = 500

# Upaya menimpa prompt sistem / prompt injection.
_INJECTION_PATTERNS = (
    r'abaikan', r'lupakan\s+(?:semua\s+)?aturan', r'tanpa\s+(?:batasan|aturan)',
    r'instruksi\s+sebelumnya', r'prompt\s+sistem', r'system\s+prompt',
    r'ignore', r'disregard', r'previous\s+instructions', r'no\s+(?:rules|restrictions)',
    r'jailbreak', r'developer\s+mode',
)
# Upaya melewati prinsip pedagogis (jawaban akhir tanpa proses).
_PEDAGOGY_BYPASS_PATTERNS = (
    r'kunci\s+jawaban', r'jawaban\s+(?:akhir|langsung)\s+saja', r'langsung\s+(?:kasih\s+|beri\s+|berikan\s+)?jawab(?:an)?',
    r'tanpa\s+penjelasan', r'kerjakan\s+(?:tugas|pr|soal)', r'answer\s+key',
    r'just\s+(?:give|tell)\s+(?:me\s+)?the\s+answer', r'without\s+(?:any\s+)?explanation',
    r'do\s+my\s+homework',
)
# Gaya yang tidak pantas untuk siswa SMA.
_INAPPROPRIATE_PATTERNS = (
    r'kasar', r'(?:meng)?hina', r'maki', r'umpat', r'bodoh', r'goblok', r'tolol',
    r'anjing', r'bangsat', r'seksi', r'seksual', r'mesum', r'pacar', r'romantis',
    r'insult', r'stupid', r'idiot', r'swear', r'curse', r'sexy', r'sexual', r'nsfw',
    r'girlfriend', r'boyfriend', r'romantic', r'flirt',
)
_RULES = (
    ('persona_injection', _INJECTION_PATTERNS),
    ('persona_bypasses_pedagogy', _PEDAGOGY_BYPASS_PATTERNS),
    ('persona_inappropriate', _INAPPROPRIATE_PATTERNS),
)
_MESSAGES = {
    'persona_too_long': f'Persona maksimal {MAX_PERSONA_CHARS} karakter.',
    'persona_injection': 'Persona tidak boleh berisi perintah untuk mengubah atau mengabaikan aturan AI.',
    'persona_bypasses_pedagogy': 'Persona tidak boleh meminta jawaban akhir tanpa proses berpikir.',
    'persona_inappropriate': 'Persona tidak boleh berisi gaya kasar, menghina, atau romantis/seksual.',
}


class PersonaError(ValueError):
    def __init__(self, reason):
        super().__init__(_MESSAGES[reason])
        self.reason = reason


def clean_persona(text):
    """Return normalized persona text, or raise PersonaError. Empty text = no persona."""
    text = unicodedata.normalize('NFKC', text or '')
    text = ''.join(ch if ch.isprintable() or ch == '\n' else ' ' for ch in text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    # Cegah persona menutup tag pembungkusnya sendiri.
    text = text.replace('<', '‹').replace('>', '›')
    if len(text) > MAX_PERSONA_CHARS:
        raise PersonaError('persona_too_long')
    lowered = text.casefold()
    for reason, patterns in _RULES:
        if any(re.search(rf'\b{pattern}\b', lowered) for pattern in patterns):
            raise PersonaError(reason)
    return text
