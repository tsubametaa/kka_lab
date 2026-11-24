"""Automatic detection and extraction of natural-language persona instructions from chat.

Mendukung deteksi otomatis saat siswa mengetik instruksi persona langsung di kolom chat
(mis. "mulai sekarang panggil aku ntik karena aku femboy cantik", "jawab pakai gaya
tsundere", atau "hapus persona"), tanpa harus memakai tombol/perintah /persona khusus.

Arsitektur 2-tahap (Hybrid):
1. Stage 1 (Deterministik/Cepat): Pola heuristik ID/EN presisi tinggi + guard ketat
   anti-false-positive (tidak salah tangkap pertanyaan teori KKA atau prompt tugas panjang).
2. Stage 2 (Opsional LLM Verifier/Extractor): Dipicu hanya ketika pesan pendek memuat
   sinyal ambigu namun belum tertangkap Stage 1; memakai greedy decoding + JSON schema.
Semua persona hasil ekstraksi WAJIB lolos `clean_persona()` (safety guardrails).
"""

from dataclasses import dataclass
import json
import os
import re
import unicodedata

from app.features.language.slang import has_slang, normalize_slang
from app.features.persona.validator import MAX_PERSONA_CHARS, PersonaError, clean_persona

MAX_AUTO_PERSONA_INPUT_CHARS = 360

_QUESTION_ABOUT_PERSONA_RE = re.compile(
    r'^\s*(?:apa(?:kah)?|jelaskan|sebutkan|uraikan|bagaimana|gimana|mengapa|kenapa|'
    r'bandingkan|berikan\s+contoh|contoh|pengertian|definisi|arti(?:nya)?|maksud(?:nya)?|'
    r'what(?:\s+is|\s+are|\'s)?|how\s+(?:does|do|to|is)|why\s+(?:is|are|do|does)|'
    r'explain|define|describe|compare|give\s+(?:an?\s+)?example)\b',
    flags=re.IGNORECASE,
)

# Permintaan ganti bahasa saja sudah ditangani detector bahasa, bukan persona.
_LANGUAGE_ONLY_RE = re.compile(
    r'^\s*(?:mulai\s+sekarang\s*,?\s*|tolong\s+|please\s+|from\s+now\s+on\s*,?\s*)?'
    r'(?:jawab(?:lah)?|balas(?:lah)?|jelaskan|bicara(?:lah)?|gunakan|pakai|'
    r'answer|reply|respond|explain|speak|use)\s+'
    r'(?:dalam\s+|dengan\s+|pakai\s+|menggunakan\s+|in\s+|using\s+)?'
    r'(?:bahasa\s+(?:indonesia|inggris)|indonesian|english)\s*(?:ya|aja|saja|dong|please)?[.!?]*\s*$',
    flags=re.IGNORECASE,
)

# Indikator prompt tugas panjang/kurikulum (bukan preferensi persona permanen).
_LONG_TASK_PROMPT_RE = re.compile(
    r'\b(?:buat(?:kan|lah)?\s+(?:materi|soal|rangkuman|rencana|silabus|modul|artikel|esai|kode|program)|'
    r'susun(?:lah)?\s+(?:materi|soal|modul|rencana)|'
    r'tujuan\s+pembelajaran|evaluasi\s+pembelajaran|pilihan\s+ganda|ketentuan\s+penyusunan|'
    r'create\s+(?:a\s+)?(?:lesson|curriculum|quiz|exam|module|essay|article))\b',
    flags=re.IGNORECASE,
)

# Perintah reset / hapus persona dalam bahasa alami (ID & EN).
_RESET_PATTERNS = (
    re.compile(
        r'^\s*(?:mulai\s+sekarang\s*,?\s*|dari\s+sekarang\s*,?\s*)?'
        r'(?:tolong\s+|coba\s+|boleh\s+)?'
        r'(?:hapus|reset|matikan|nonaktifkan|hilangkan|cabut|batalkan|lepas|stop|hentikan)\s+'
        r'(?:semua\s+)?(?:persona|karakter|gaya\s+(?:bicara|bahasa|jawaban|menjawab|ngobrol))'
        r'(?:\s+(?:kamu|mu|anda|ai|ini|sekarang|aja|saja|ya|dong))*\s*[.!?]*$',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:mulai\s+sekarang\s*,?\s*|dari\s+sekarang\s*,?\s*)?'
        r'(?:tolong\s+|coba\s+)?'
        r'(?:kembali|balik)\s+(?:lagi\s+)?(?:ke\s+)?(?:gaya|mode|persona|bicara|sifat)?\s*'
        r'(?:normal|biasa|standar|default)(?:\s+(?:aja|saja|lagi|ya|dong))*\s*[.!?]*$',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:mulai\s+sekarang\s*,?\s*)?'
        r'(?:jangan|enggak\s+usah|gak\s+usah|nggak\s+usah|tak\s+usah|berhenti)\s+'
        r'(?:pakai|pake|gunakan|menggunakan)\s+(?:persona|gaya\s+(?:bicara|bahasa|itu|tadi))'
        r'(?:\s+(?:lagi|ya|dong|aja|saja))*\s*[.!?]*$',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:from\s+now\s+on\s*,?\s*)?(?:please\s+)?'
        r'(?:reset|clear|remove|delete|disable|turn\s+off|stop\s+using)\s+'
        r'(?:the\s+|your\s+|my\s+)?(?:persona|character|speaking\s+style|tone)'
        r'(?:\s+(?:now|please))*\s*[.!?]*$',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:from\s+now\s+on\s*,?\s*)?(?:please\s+)?'
        r'(?:go\s+back|return|switch\s+back)\s+to\s+(?:normal|default|regular)\s*'
        r'(?:mode|style|tone|persona)?\s*[.!?]*$',
        flags=re.IGNORECASE,
    ),
)

# Pola eksplisit pemasangan persona (ID & EN).
# Setiap tuple: (compiled_regex, extract_mode)
# extract_mode:
# - 'group': ambil group('body') sebagai teks persona
# - 'whole': simpan kalimat instruksi yang bersih (menghapus filler pembuka saja)
_SET_PATTERNS = (
    # 1. Explicit meta-command (ID & EN): "pasang/atur/set/ubah/ganti persona jadi/to ..."
    (
        re.compile(
            r'^\s*(?:mulai\s+sekarang\s*,?\s*|dari\s+sekarang\s*,?\s*|from\s+now\s+on\s*,?\s*)?'
            r'(?:tolong\s+|coba\s+|boleh\s+|please\s+|aku\s+mau\s+|saya\s+mau\s+)?'
            r'(?:pasang(?:kan)?|atur|set(?:ting)?|ubah|ganti|aktifkan|pakai|pake|gunakan|'
            r'change|update|switch|use)\s+'
            r'(?:your\s+|my\s+|the\s+)?(?:persona|karakter)\s*'
            r'(?:kamu|mu|anda|ai|asisten|saya|aku|ku)?\s*'
            r'(?:menjadi|jadi|ke|sebagai|seperti|kayak|ala|to|as|like|:)?\s*(?P<body>.+)$',
            flags=re.IGNORECASE,
        ),
        'group',
    ),
    # 3. Calling / addressing user: "mulai sekarang panggil aku ntik karena aku femboy cantik"
    (
        re.compile(
            r'^\s*(?:(?:mulai|dari)\s+sekarang|untuk\s+seterusnya|seterusnya|ke\s+depan(?:nya)?)\s*,?\s*'
            r'(?:tolong\s+|coba\s+|boleh(?:kah)?\s+|bisa(?:kah)?\s+)?'
            r'(?:(?:kamu|kau|anda)\s+)?'
            r'(?:panggil|sebut|sapa)\s+(?:aku|saya|gue|gw|diriku|namaku)\b.*$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    (
        re.compile(
            r'^\s*(?:tolong\s+|coba\s+|boleh(?:kah)?\s+|bisa(?:kah)?\s+)?'
            r'(?:(?:kamu|kau|anda)\s+)?'
            r'(?:panggil|sebut|sapa)\s+(?:aku|saya|gue|gw|diriku)\s+'
            r'(?:dengan\s+|sebagai\s+|pakai\s+|pake\s+)?\S.*$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 4. Calling / addressing user (EN): "from now on call me Alex ..."
    (
        re.compile(
            r'^\s*(?:from\s+now\s+on\s*,?\s*)?(?:please\s+|can\s+you\s+|could\s+you\s+)?'
            r'(?:call|address)\s+me\s+\S.*$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 5. "mulai sekarang <kamu/jawab/pakai gaya/jadi ...>"
    (
        re.compile(
            r'^\s*(?:(?:mulai|dari)\s+sekarang|untuk\s+seterusnya|seterusnya)\s*,?\s*'
            r'(?:tolong\s+|coba\s+)?'
            r'(?:(?:kamu|kau|anda|ai|asisten)\s+)?'
            r'(?:pakai|pake|gunakan|menggunakan|jawab(?:lah)?|balas(?:lah)?|bicara(?:lah)?|ngomong(?:lah)?|'
            r'berperan(?:lah)?|bersikap(?:lah)?|bertindak(?:lah)?|jadilah|jadi|ubah|ganti|akhiri|awali|'
            r'abaikan|lupakan|langsung|kasih|berikan|beri)\b.+$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 6. "from now on <you/speak/talk/act/be/answer ...>"
    (
        re.compile(
            r'^\s*from\s+now\s+on\s*,?\s*(?:please\s+)?'
            r'(?:(?:you\s+)?(?:speak|talk|reply|respond|answer|act|behave|be|use|adopt|roleplay|ignore|disregard|just)\b).+$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 7. Style / tone instruction ID: "pakai/gunakan/ubah/ganti gaya bicara/bahasa/jawaban ..."
    (
        re.compile(
            r'^\s*(?:tolong\s+|coba\s+|boleh\s+|kamu\s+)?'
            r'(?:pakai|pake|gunakan|menggunakan|ubah|ganti|atur|jawab(?:lah)?\s+(?:dengan|pakai|pake|menggunakan)|'
            r'bicara(?:lah)?\s+(?:dengan|pakai|pake|menggunakan)|ngomong(?:lah)?\s+(?:dengan|pakai|pake|menggunakan))\s+'
            r'(?:gaya(?:\s+(?:bicara|bahasa|jawaban|menjawab|ngobrol))?|nada(?:\s+(?:bicara|bahasa))?|logat|aksen)\b.+$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 8. "aku/saya ingin/mau kamu/AI ini menjawab/berperan/memanggil ..."
    (
        re.compile(
            r'^\s*(?:aku|saya|gue|gw)\s+(?:ingin|mau|pengen|pingin)\s+'
            r'(?:kamu|kau|anda|ai(?:\s+ini)?|asisten(?:\s+ini)?)\s+'
            r'(?:menjawab|jawab|membalas|balas|berbicara|bicara|ngomong|berperan|bersikap|bertindak|'
            r'jadi|menjadi|memanggil|panggil|menyebut|sebut|pakai\s+gaya|menggunakan\s+gaya)\b.+$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 9. Direct role/style imperative ID: "jawablah/bicaralah/jadilah/berperanlah seperti/kayak/ala ..."
    (
        re.compile(
            r'^\s*(?:tolong\s+|coba\s+)?'
            r'(?:(?:jawab(?:lah)?|balas(?:lah)?|bicara(?:lah)?|ngomong(?:lah)?|berperan(?:lah)?|bersikap(?:lah)?)\s+'
            r'(?:seperti|kayak|layaknya|selayaknya|ala|sebagai)|'
            r'(?:jadilah|bertindaklah\s+sebagai)\s+)'
            r'.+$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
    # 10. Direct style/role imperative EN: "speak/talk/answer/act like/as ..."
    (
        re.compile(
            r'^\s*(?:please\s+)?'
            r'(?:(?:speak|talk|reply|respond|answer)\s+(?:like|as|in\s+(?:a|an|the)\s+\S+(?:\s+\S+){0,4}\s+(?:style|tone|voice|manner|persona))|'
            r'(?:act|behave|roleplay)\s+(?:as|like)|be\s+(?:my|a|an))\s+.+$',
            flags=re.IGNORECASE,
        ),
        'whole',
    ),
)

# Kata pembuka perintah waktu/sopan santun yang dirapikan agar deskripsi persona bersih.
_LEADING_COMMAND_PREFIX_RE = re.compile(
    r'^\s*(?:(?:mulai|dari)\s+sekarang|untuk\s+seterusnya|seterusnya|ke\s+depan(?:nya)?|'
    r'from\s+now\s+on)\s*,?\s*',
    flags=re.IGNORECASE,
)
_LEADING_POLITE_PREFIX_RE = re.compile(
    r'^\s*(?:tolong|coba|mohon|please|can\s+you|could\s+you|'
    r'(?:boleh(?:kah)?|bisa(?:kah)?)(?:\s+(?:kamu|kau|anda))?)\s+',
    flags=re.IGNORECASE,
)

# Sinyal lemah untuk menentukan kapan Stage 2 (LLM) layak dipanggil.
_WEAK_PERSONA_SIGNAL_RE = re.compile(
    r'\b(?:persona|karakter|tsundere|kuudere|yandere|wibu|bajak\s+laut|pirate|'
    r'panggil\s+(?:aku|saya|gue|gw)|sebut\s+(?:aku|saya|gue|gw)|call\s+me|address\s+me|'
    r'mulai\s+sekarang|dari\s+sekarang|seterusnya|from\s+now\s+on|'
    r'gaya\s+(?:bicara|bahasa|jawaban|menjawab|ngobrol)|nada\s+bicara|speaking\s+style|'
    r'akhiri\s+(?:jawaban|kalimat|setiap)|sapa\s+(?:aku|saya)|'
    r'kayak\s+(?:teman|temen|sahabat|kakak|adik|anime|robot|guru|dosen)|'
    r'seperti\s+(?:teman|temen|sahabat|kakak|adik|anime|robot)|'
    r'roleplay|berperan)\b',
    flags=re.IGNORECASE,
)

# Pemisah antara kalimat instruksi persona dan pertanyaan lanjutan dalam satu pesan.
_CLAUSE_SPLIT_RE = re.compile(r'(?:\n+|(?<=[.!?])\s+)')
_CONJUNCTION_QUESTION_RE = re.compile(
    r'\s*(?:,|;|\b(?:dan|terus|trus|lalu|nah|sekarang|and|now)\b)\s+'
    r'(?=(?:apa(?:kah)?|bagaimana|gimana|mengapa|kenapa|siapa|kapan|dimana|jelaskan|sebutkan|'
    r'uraikan|bandingkan|tolong\s+jelaskan|coba\s+jelaskan|what|how|why|who|when|where|explain|define|compare)\b)',
    flags=re.IGNORECASE,
)

_LLM_EXTRACTOR_SYSTEM_PROMPT = (
    'You are an intent classifier for a Grade 10 KKA tutoring chatbot. '
    'Decide if the student message is asking to SET or RESET the assistant\'s ongoing speaking persona, '
    'nickname/form of address (e.g., "call me ..."), tone, catchphrase, or character style.\n'
    'Return ONLY valid JSON with exact keys:\n'
    '{"action": "set" | "reset" | "none", "persona": "string", "remainder": "string"}\n'
    'Rules:\n'
    '- "set": The user wants the assistant to adopt a speaking style, character, catchphrase, or call the user by a specific name/nickname. '
    'Put the clean persona instruction in "persona". If the message ALSO includes a separate curriculum question, put ONLY that question in "remainder", otherwise "".\n'
    '- "reset": The user wants to remove/reset the persona back to normal. Set "persona" to "".\n'
    '- "none": Normal curriculum questions, questions ABOUT personas/prompting, language-only requests, greetings, or long one-off assignment prompts. Set "persona" and "remainder" to "".\n'
    '- Never answer the user\'s question. Return ONLY the JSON object.'
)


@dataclass(frozen=True)
class PersonaCommand:
    """Result of checking a chat message for automatic persona set/reset intent."""

    action: str = 'none'  # 'none' | 'set' | 'reset' | 'rejected'
    persona: str = ''
    remainder: str = ''
    reason: str = ''
    message: str = ''
    source: str = ''  # 'heuristic' | 'llm' | ''

    @property
    def is_persona_command(self) -> bool:
        return self.action in ('set', 'reset', 'rejected')


def is_auto_persona_enabled() -> bool:
    """Feature flag for automatic persona detection from chat (enabled by default)."""
    val = os.getenv('AUTO_PERSONA_ENABLED', 'true').strip().lower()
    return val not in ('0', 'false', 'no', 'off')


def is_auto_persona_llm_enabled() -> bool:
    """Feature flag for Stage-2 LLM persona extraction on ambiguous short messages."""
    val = os.getenv('AUTO_PERSONA_LLM_ENABLED', 'true').strip().lower()
    return val not in ('0', 'false', 'no', 'off')


def _normalize_input(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text or '')
    normalized = ''.join(ch if ch.isprintable() or ch == '\n' else ' ' for ch in normalized)
    return normalized.strip()


def _looks_like_long_task_prompt(text: str) -> bool:
    """Return True for multi-paragraph assignment/curriculum role-prompts, not chat personas."""
    if len(text) > MAX_AUTO_PERSONA_INPUT_CHARS:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4:
        return True
    numbered_lines = sum(1 for line in lines if re.match(r'^\d+[.)]\s+', line))
    if numbered_lines >= 2:
        return True
    if _LONG_TASK_PROMPT_RE.search(text) and len(text) > 140:
        return True
    return False


def _looks_like_question(text: str) -> bool:
    stripped = (text or '').strip()
    if not stripped:
        return False
    if stripped.endswith('?'):
        return True
    return bool(_QUESTION_ABOUT_PERSONA_RE.match(stripped))


def _split_instruction_and_remainder(text: str) -> tuple[str, str]:
    """Split a message into (candidate_persona_clause, optional_followup_question)."""
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) >= 2:
        first = parts[0]
        rest = ' '.join(parts[1:]).strip()
        # Jika bagian kedua terlihat seperti pertanyaan / permintaan materi, pisahkan sebagai remainder.
        if _looks_like_question(rest) or _LONG_TASK_PROMPT_RE.search(rest):
            return first, rest
        # Jika kalimat pertama sendiri sudah reset command, sisanya selalu remainder.
        if any(pat.match(first) for pat in _RESET_PATTERNS):
            return first, rest
        return text, ''

    # Satu kalimat, cek apakah digabung dengan koma + kata tanya ("panggil aku Budi, apa itu AI?")
    split_match = _CONJUNCTION_QUESTION_RE.search(text)
    if split_match and split_match.start() >= 8:
        left = text[:split_match.start()].strip(' ,;')
        right = text[split_match.end():].strip()
        if left and right:
            return left, right
    return text, ''


def _polish_persona_text(raw: str, strip_time_prefix: bool = True) -> str:
    cleaned = (raw or '').strip(' "\'`')
    if strip_time_prefix:
        cleaned = _LEADING_COMMAND_PREFIX_RE.sub('', cleaned).strip()
        cleaned = _LEADING_POLITE_PREFIX_RE.sub('', cleaned).strip()
    cleaned = cleaned.strip(' ,:;-?.!')
    # Rapikan spasi ganda sambil menjaga nuansa gaya bicara siswa.
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _validate_extracted_persona(raw_persona: str, remainder: str, source: str) -> PersonaCommand:
    if not raw_persona or len(raw_persona) < 2:
        return PersonaCommand(action='none')
    try:
        validated = clean_persona(raw_persona)
    except PersonaError as err:
        return PersonaCommand(
            action='rejected',
            persona='',
            remainder=remainder,
            reason=err.reason,
            message=str(err),
            source=source,
        )
    if not validated:
        return PersonaCommand(action='none')
    return PersonaCommand(
        action='set',
        persona=validated,
        remainder=remainder,
        source=source,
    )


def _detect_persona_command_single(cleaned_input: str) -> PersonaCommand:
    if not cleaned_input:
        return PersonaCommand(action='none')
    if _looks_like_long_task_prompt(cleaned_input):
        return PersonaCommand(action='none')
    if _LANGUAGE_ONLY_RE.match(cleaned_input):
        return PersonaCommand(action='none')

    # Pertanyaan murni tentang konsep/materi (mis. "Apa itu persona dalam prompt engineering?")
    if _QUESTION_ABOUT_PERSONA_RE.match(cleaned_input):
        return PersonaCommand(action='none')

    candidate, remainder = _split_instruction_and_remainder(cleaned_input)

    # Jika bagian kandidat ternyata pertanyaan (bukan instruksi), lewati.
    if _QUESTION_ABOUT_PERSONA_RE.match(candidate):
        return PersonaCommand(action='none')
    if _LANGUAGE_ONLY_RE.match(candidate):
        return PersonaCommand(action='none')

    # Cek perintah reset/hapus persona terlebih dahulu.
    for reset_pat in _RESET_PATTERNS:
        if reset_pat.match(candidate):
            return PersonaCommand(
                action='reset',
                persona='',
                remainder=remainder,
                source='heuristic',
            )

    # Cek pola pemasangan persona.
    for set_pat, mode in _SET_PATTERNS:
        match = set_pat.match(candidate)
        if not match:
            continue
        if mode == 'group':
            body = _polish_persona_text(match.group('body'), strip_time_prefix=False)
            # Hindari menangkap kata reset yang lolos ke pola ubah/ganti.
            if body.casefold() in ('normal', 'biasa', 'standar', 'default', 'kosong', 'off', 'none', 'hapus', 'reset'):
                return PersonaCommand(
                    action='reset',
                    persona='',
                    remainder=remainder,
                    source='heuristic',
                )
            return _validate_extracted_persona(body, remainder, source='heuristic')

        polished = _polish_persona_text(candidate, strip_time_prefix=True)
        return _validate_extracted_persona(polished, remainder, source='heuristic')

    return PersonaCommand(action='none')


def detect_persona_command(text: str) -> PersonaCommand:
    """Stage 1: Fast deterministic detection of persona set/reset instructions from chat."""
    cleaned_input = _normalize_input(text)
    cmd = _detect_persona_command_single(cleaned_input)
    if cmd.is_persona_command:
        return cmd
    if has_slang(cleaned_input):
        return _detect_persona_command_single(_normalize_input(normalize_slang(cleaned_input)))
    return PersonaCommand(action='none')


def might_be_persona_instruction(text: str) -> bool:
    """Return True if a short message has weak persona signals worth checking via Stage-2 LLM."""
    cleaned_input = _normalize_input(text)
    if not cleaned_input or _looks_like_long_task_prompt(cleaned_input):
        return False
    expanded = _normalize_input(normalize_slang(cleaned_input))
    if _LANGUAGE_ONLY_RE.match(expanded):
        return False
    if _QUESTION_ABOUT_PERSONA_RE.match(expanded):
        return False
    return bool(_WEAK_PERSONA_SIGNAL_RE.search(cleaned_input) or _WEAK_PERSONA_SIGNAL_RE.search(expanded))


def build_persona_extractor_messages(text: str) -> list[dict[str, str]]:
    """Build chat messages for Stage-2 LLM persona intent classification & slot extraction."""
    return [
        {'role': 'system', 'content': _LLM_EXTRACTOR_SYSTEM_PROMPT},
        {'role': 'user', 'content': _normalize_input(text)[:MAX_AUTO_PERSONA_INPUT_CHARS]},
    ]


def parse_llm_persona_command(raw_output: str, original_text: str = '') -> PersonaCommand:
    """Parse and validate JSON returned by the Stage-2 LLM persona extractor."""
    if not raw_output:
        return PersonaCommand(action='none')
    cleaned = re.sub(
        r'^\s*```(?:json)?\s*|\s*```\s*$', '', raw_output.strip(), flags=re.IGNORECASE
    )
    # Ekstrak blok JSON pertama jika model menambahkan teks tipis di sekitarnya.
    brace_match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if brace_match:
        cleaned = brace_match.group(0)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return PersonaCommand(action='none')
    if not isinstance(data, dict):
        return PersonaCommand(action='none')

    action = str(data.get('action') or 'none').strip().casefold()
    remainder = _normalize_input(str(data.get('remainder') or ''))
    if remainder == _normalize_input(original_text):
        remainder = ''

    if action == 'reset':
        return PersonaCommand(action='reset', persona='', remainder=remainder, source='llm')
    if action == 'set':
        raw_persona = _polish_persona_text(str(data.get('persona') or ''), strip_time_prefix=True)
        if not raw_persona:
            return PersonaCommand(action='none')
        if len(raw_persona) > MAX_PERSONA_CHARS:
            return PersonaCommand(
                action='rejected',
                reason='persona_too_long',
                message=str(PersonaError('persona_too_long')),
                remainder=remainder,
                source='llm',
            )
        return _validate_extracted_persona(raw_persona, remainder, source='llm')
    return PersonaCommand(action='none')


def resolve_persona_command(text: str, assistant=None) -> PersonaCommand:
    """Synchronous hybrid persona command resolution (Stage 1 heuristic -> optional Stage 2 LLM)."""
    if not is_auto_persona_enabled():
        return PersonaCommand(action='none')
    cmd = detect_persona_command(text)
    if cmd.is_persona_command:
        return cmd
    if (
        assistant is not None
        and is_auto_persona_llm_enabled()
        and might_be_persona_instruction(text)
        and hasattr(assistant, 'extract_persona')
    ):
        try:
            raw = assistant.extract_persona(text)
            return parse_llm_persona_command(raw, original_text=text)
        except Exception:
            return PersonaCommand(action='none')
    return PersonaCommand(action='none')


async def aresolve_persona_command(text: str, assistant=None) -> PersonaCommand:
    """Async hybrid persona command resolution for FastAPI routes."""
    if not is_auto_persona_enabled():
        return PersonaCommand(action='none')
    cmd = detect_persona_command(text)
    if cmd.is_persona_command:
        return cmd
    if assistant is None or not is_auto_persona_llm_enabled() or not might_be_persona_instruction(text):
        return PersonaCommand(action='none')
    try:
        if hasattr(assistant, 'aextract_persona'):
            raw = await assistant.aextract_persona(text)
            return parse_llm_persona_command(raw, original_text=text)
        if hasattr(assistant, 'extract_persona'):
            import asyncio
            raw = await asyncio.to_thread(assistant.extract_persona, text)
            return parse_llm_persona_command(raw, original_text=text)
    except Exception:
        return PersonaCommand(action='none')
    return PersonaCommand(action='none')


def build_persona_confirmation_context(command: PersonaCommand, lang: str = 'id') -> str:
    """Build synthetic system context so LLM confirms persona changes naturally without RAG."""
    if command.action == 'reset':
        if lang == 'en':
            return (
                'System Note: The student just reset their custom persona back to the default style, '
                'and the change has been saved automatically. Confirm briefly (1-2 warm sentences) that '
                'your speaking style is back to normal, and ask what KKA topic they want to learn. '
                'Do not say material is missing and do not tell them to ask the teacher.'
            )
        return (
            'Catatan Sistem: Siswa baru saja menghapus/mereset persona khusus ke gaya normal dan '
            'perubahan ini sudah tersimpan otomatis. Konfirmasi secara singkat dan ramah (1-2 kalimat) '
            'bahwa gaya bicaramu sudah kembali normal, lalu tanyakan materi KKA apa yang ingin dipelajari. '
            'Jangan menyatakan materi tidak ditemukan dan jangan menyarankan bertanya kepada guru.'
        )

    if lang == 'en':
        return (
            'System Note: The student just updated their preferred persona/form of address and it has '
            f'been saved automatically as: "{command.persona}". '
            'Confirm briefly and warmly (1-2 sentences) that this persona is now active for future messages, '
            'while IMMEDIATELY adopting the requested tone, style, or nickname from the very first word. '
            'Do not say material is missing, do not lecture on curriculum topics yet, and do not tell them to ask the teacher. '
            'Simply invite them to ask a KKA topic whenever ready.'
        )
    return (
        'Catatan Sistem: Siswa baru saja mengatur preferensi persona/sapaan baru dan pengaturan ini '
        f'sudah tersimpan otomatis sebagai: "{command.persona}". '
        'Konfirmasi secara singkat dan alami (1-2 kalimat) bahwa persona tersebut sudah aktif untuk percakapan '
        'selanjutnya, sambil LANGSUNG menerapkan sapaan, nada, dan gaya bicara persona tersebut sejak kata pertama. '
        'Jangan menyatakan materi tidak ditemukan, jangan menjelaskan materi pelajaran dulu, dan jangan menyarankan '
        'bertanya kepada guru. Cukup tanyakan materi KKA apa yang ingin dibahas.'
    )


def build_persona_rejection_reply(command: PersonaCommand, lang: str = 'id') -> str:
    """Return a polite refusal when an auto-detected persona fails safety validation."""
    detail = command.message or 'Persona tidak memenuhi aturan keamanan.'
    if lang == 'en':
        return (
            f'Sorry, I cannot apply that persona setting: {detail} '
            'Please choose another friendly, respectful speaking style or nickname.'
        )
    return (
        f'Maaf, pengaturan persona tersebut tidak dapat diterapkan: {detail} '
        'Kamu bisa memilih gaya bicara atau sapaan lain yang tetap ramah dan sopan ya.'
    )
