"""Local language detection, prompt formatting reminder, and mismatch detection."""

import re

from app.features.language.locales import LOCALES, en, id

_DEFINITION_QUERY_RE = re.compile(
    r'\b(?:means?|meaning|stands?\s+for|arti(?:nya)?|kepanjangan(?:nya)?|singkatan)\b',
    flags=re.IGNORECASE,
)
_EXPLICIT_EN_RE = re.compile(
    r'\b(?:jawab|jelaskan|balas|terjemahkan|pakai|gunakan|'
    r'answer|reply|respond|explain|translate|speak|use)\b'
    r'(?P<span_en>[^?.!\n]{0,80}?)\b(?:dalam\s+|pakai\s+|menggunakan\s+|ke\s+|in\s+|into\s+|using\s+)?'
    r'(?:bahasa\s+inggris|english)\b',
    flags=re.IGNORECASE,
)
_EXPLICIT_ID_RE = re.compile(
    r'\b(?:jawab|jelaskan|balas|terjemahkan|pakai|gunakan|'
    r'answer|reply|respond|explain|translate|speak|use)\b'
    r'(?P<span_id>[^?.!\n]{0,80}?)\b(?:dalam\s+|pakai\s+|menggunakan\s+|ke\s+|in\s+|into\s+|using\s+)?'
    r'(?:bahasa\s+indonesia|indonesian)\b',
    flags=re.IGNORECASE,
)
_EN_OPENER_RE = re.compile(
    r'^\s*(?:what|how|why|who|whom|whose|when|where|which|'
    r'do|does|did|is|are|am|was|were|can|could|would|should|will|shall|may|might|'
    r'have|has|please|explain|describe|tell|give|show|define|compare|summarize|'
    r'help|hello|hi|hey)\b',
    flags=re.IGNORECASE,
)
_ID_OPENER_RE = re.compile(
    r'^\s*(?:apa|apakah|bagaimana|gimana|mengapa|kenapa|siapa|kapan|'
    r'dimana|di\s+mana|kemana|mana|berapa|jelaskan|sebutkan|uraikan|'
    r'bandingkan|ringkas|tolong|mohon|coba|bisa|bisakah|boleh|bolehkah|'
    r'apabila|kalau|jika|halo|hai|selamat)\b',
    flags=re.IGNORECASE,
)
_EN_STANDALONE_NO_RE = re.compile(
    r'^\s*no(?:[,!.\s]*$|\s+(?:thanks|thank|it|that|not)\b)',
    flags=re.IGNORECASE,
)
_EN_CONTRACTIONS = {
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "couldn't", "wouldn't", "shouldn't", "won't", "i'm", "i've",
    "i'd", "i'll", "you're", "we're", "they're", "it's", "that's",
    "what's", "how's", "who's", "where's", "let's",
}
_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ]+(?:'[a-zA-Z]+)?")


def _explicit_language_request(text):
    match_en = _EXPLICIT_EN_RE.search(text)
    if match_en and not _DEFINITION_QUERY_RE.search(match_en.group('span_en')):
        return 'en'
    match_id = _EXPLICIT_ID_RE.search(text)
    if match_id and not _DEFINITION_QUERY_RE.search(match_id.group('span_id')):
        return 'id'
    return None


def _language_scores(text):
    if not text or not text.strip():
        return 0.0, 0.0
    normalized = text.replace('’', "'").replace('`', "'").casefold()
    id_score = 0.0
    en_score = 0.0

    if _ID_OPENER_RE.search(normalized):
        id_score += 1.5
    if _EN_OPENER_RE.search(normalized):
        en_score += 1.5
    if _EN_STANDALONE_NO_RE.search(normalized):
        en_score += 1.5

    for token in _WORD_RE.findall(normalized):
        if token in _EN_CONTRACTIONS:
            en_score += 1.5
            continue
        if token in id.DETECTION_WORDS:
            id_score += 1.0
        elif len(token) >= 5 and token.endswith(('nya', 'kah')):
            id_score += 1.0
        if token in en.DETECTION_WORDS:
            en_score += 1.0
    return id_score, en_score


def detect_language(text, history=None, default='id'):
    """Detect whether the user's latest message is Indonesian ('id') or English ('en')."""
    explicit = _explicit_language_request(text or '')
    if explicit:
        return explicit

    id_score, en_score = _language_scores(text)
    if en_score > id_score:
        return 'en'
    if id_score > en_score:
        return 'id'

    for message in reversed(history or ()):
        if message.get('role') != 'user':
            continue
        content = message.get('content', '')
        prior_explicit = _explicit_language_request(content)
        if prior_explicit:
            return prior_explicit
        prior_id, prior_en = _language_scores(content)
        if prior_en != prior_id:
            return 'en' if prior_en > prior_id else 'id'

    return default if default in LOCALES else 'id'


def format_user_message(question, lang):
    """Attach a transient language reminder for model generation without altering stored history."""
    locale = LOCALES.get(lang, id)
    reminder = locale.USER_LANGUAGE_REMINDER
    stripped = question.strip()
    if stripped.endswith(reminder):
        return stripped
    return f'{stripped}\n\n{reminder}'


def is_wrong_language(text, expected_lang):
    """Return True when generated text is clearly written in the opposite language."""
    id_score, en_score = _language_scores(text)
    if expected_lang == 'en':
        return id_score >= 3.0 and id_score >= en_score * 2.0
    if expected_lang == 'id':
        return en_score >= 3.0 and en_score >= id_score * 2.0
    return False

