"""Bilingual (Indonesian & English) chat slang and abbreviation normalizer.

Menormalisasi singkatan/slang percakapan siswa (mis. "idk", "dunno", "r u know abt tht",
"smth", "skrg", "gtw", "knp", "gmn", "yg", "dgn") menjadi bentuk baku untuk keperluan
deteksi bahasa, pencarian RAG, deteksi intent persona/rekomendasi, serta bantuan konteks
prompt ke LLM — sambil melindungi blok kode, URL, dan istilah teknis KKA.
"""

import re
import unicodedata

# Segmen yang TIDAK boleh dimodifikasi: fenced code block, inline code, dan URL.
_PROTECTED_SEGMENT_RE = re.compile(
    r'(```[\s\S]*?```|`[^`\n]+`|https?://\S+)',
    flags=re.IGNORECASE,
)

# 1. Frasa slang multi-kata (diproses lebih dulu agar konteks terjaga).
_PHRASE_SLANG = {
    # English multi-word / chat phrases
    'r u know': 'do you know',
    'u know': 'you know',
    'r u': 'are you',
    'u r': 'you are',
    'do u': 'do you',
    'did u': 'did you',
    'can u': 'can you',
    'could u': 'could you',
    'would u': 'would you',
    'will u': 'will you',
    'have u': 'have you',
    'hv u': 'have you',
    'thank u': 'thank you',
    'for u': 'for you',
    'abt u': 'about you',
    'about u': 'about you',
    'with u': 'with you',
    'tell u': 'tell you',
    'ask u': 'ask you',
    'what r u': 'what are you',
    'how r u': 'how are you',
    'i dont know': "I don't know",
    'dont know': "don't know",
    'dont understand': "don't understand",
    'i dont understand': "I don't understand",
    'w/o': 'without',
    # Indonesian multi-word slang
    'ga tau': 'tidak tahu',
    'gak tau': 'tidak tahu',
    'gk tau': 'tidak tahu',
    'ngga tau': 'tidak tahu',
    'nggak tau': 'tidak tahu',
    'ga tahu': 'tidak tahu',
    'gak tahu': 'tidak tahu',
    'gk tahu': 'tidak tahu',
    'ga ngerti': 'tidak mengerti',
    'gak ngerti': 'tidak mengerti',
    'gk ngerti': 'tidak mengerti',
    'ngga ngerti': 'tidak mengerti',
    'nggak ngerti': 'tidak mengerti',
    'ga paham': 'tidak paham',
    'gak paham': 'tidak paham',
    'gk paham': 'tidak paham',
    'ngga paham': 'tidak paham',
    'nggak paham': 'tidak paham',
    'ga bisa': 'tidak bisa',
    'gak bisa': 'tidak bisa',
    'gk bisa': 'tidak bisa',
    'ga usah': 'tidak usah',
    'gak usah': 'tidak usah',
    'gk usah': 'tidak usah',
    'g usah': 'tidak usah',
}

# 2. Kata slang / singkatan tunggal (aman terhadap istilah KKA, Python, & SMA/SMK).
_WORD_SLANG = {
    # English slang & abbreviations
    'idk': "I don't know",
    'idrk': "I don't really know",
    'idts': "I don't think so",
    'idc': "I don't care",
    'dunno': "don't know",
    'wdym': 'what do you mean',
    'wbu': 'what about you',
    'hbu': 'how about you',
    'tbh': 'to be honest',
    'ngl': 'not going to lie',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'fyi': 'for your information',
    'lmk': 'let me know',
    'nvm': 'never mind',
    'ofc': 'of course',
    'fr': 'for real',
    'rn': 'right now',
    'btw': 'by the way',
    'abt': 'about',
    'tht': 'that',
    'smth': 'something',
    'smthg': 'something',
    'sth': 'something',
    'smn': 'someone',
    'nth': 'nothing',
    'wanna': 'want to',
    'gonna': 'going to',
    'gotta': 'got to',
    'kinda': 'kind of',
    'sorta': 'sort of',
    'lemme': 'let me',
    'gimme': 'give me',
    'cuz': 'because',
    'bcuz': 'because',
    'bcz': 'because',
    'pls': 'please',
    'plz': 'please',
    'plss': 'please',
    'thx': 'thanks',
    'tysm': 'thank you so much',
    'tyvm': 'thank you very much',
    'ty': 'thank you',
    'ur': 'your',
    'b4': 'before',
    '2day': 'today',
    'tmr': 'tomorrow',
    'tmrw': 'tomorrow',
    'gr8': 'great',
    'wat': 'what',
    'wht': 'what',
    'whn': 'when',
    'whr': 'where',
    'ppl': 'people',
    'rlly': 'really',
    'rly': 'really',
    'prolly': 'probably',
    'srsly': 'seriously',
    'diff': 'difference',
    'hw': 'homework',
    'msg': 'message',
    'u': 'you',
    # Indonesian slang & abbreviations
    'skrg': 'sekarang',
    'skrng': 'sekarang',
    'skg': 'sekarang',
    'gtw': 'tidak tahu',
    'gtau': 'tidak tahu',
    'gatau': 'tidak tahu',
    'knp': 'kenapa',
    'knpa': 'kenapa',
    'knapa': 'kenapa',
    'gmn': 'bagaimana',
    'gmna': 'bagaimana',
    'gmana': 'bagaimana',
    'yg': 'yang',
    'yng': 'yang',
    'dgn': 'dengan',
    'dg': 'dengan',
    'tdk': 'tidak',
    'ga': 'tidak',
    'gk': 'tidak',
    'gak': 'tidak',
    'ngga': 'tidak',
    'nggak': 'tidak',
    'ndak': 'tidak',
    'blm': 'belum',
    'blom': 'belum',
    'udh': 'sudah',
    'udah': 'sudah',
    'sdh': 'sudah',
    'bgt': 'banget',
    'bgtt': 'banget',
    'bkn': 'bukan',
    'bknnya': 'bukannya',
    'krn': 'karena',
    'krna': 'karena',
    'karna': 'karena',
    'tp': 'tapi',
    'tpi': 'tapi',
    'dpt': 'dapat',
    'bs': 'bisa',
    'bsa': 'bisa',
    'jd': 'jadi',
    'jdi': 'jadi',
    'kl': 'kalau',
    'klo': 'kalau',
    'klu': 'kalau',
    'kalo': 'kalau',
    'org': 'orang',
    'orng': 'orang',
    'emg': 'memang',
    'emng': 'memang',
    'sbnrnya': 'sebenarnya',
    'sbnernya': 'sebenarnya',
    'sebenernya': 'sebenarnya',
    'utk': 'untuk',
    'untk': 'untuk',
    'dr': 'dari',
    'dri': 'dari',
    'sy': 'saya',
    'sya': 'saya',
    'aq': 'aku',
    'gw': 'aku',
    'gue': 'aku',
    'km': 'kamu',
    'kmu': 'kamu',
    'lu': 'kamu',
    'lo': 'kamu',
    'loe': 'kamu',
    'jg': 'juga',
    'jga': 'juga',
    'bbrp': 'beberapa',
    'msh': 'masih',
    'msih': 'masih',
    'trs': 'terus',
    'trus': 'terus',
    'gt': 'begitu',
    'gtu': 'begitu',
    'bgtu': 'begitu',
    'kyk': 'seperti',
    'kyak': 'seperti',
    'tlg': 'tolong',
    'tlng': 'tolong',
    'mksh': 'terima kasih',
    'mksih': 'terima kasih',
    'makasi': 'terima kasih',
    'tq': 'terima kasih',
    'dlu': 'dulu',
    'lg': 'lagi',
    'lgi': 'lagi',
    'bwt': 'buat',
    'dlm': 'dalam',
    'ttg': 'tentang',
    'tntg': 'tentang',
    'sbg': 'sebagai',
    'spy': 'supaya',
    'agr': 'agar',
    'hrs': 'harus',
    'prlu': 'perlu',
    'mngkn': 'mungkin',
    'mgkn': 'mungkin',
    'brp': 'berapa',
    'dmn': 'dimana',
    'dmna': 'dimana',
    'kpn': 'kapan',
    'apkh': 'apakah',
    'jwb': 'jawab',
    'jlsin': 'jelaskan',
    'jelasin': 'jelaskan',
    'cth': 'contoh',
    'cntoh': 'contoh',
    'mhs': 'mahasiswa',
}

# Gabungkan frasa (urutan terpanjang dulu) untuk single-pass re.Pattern.sub.
_PHRASE_RE = re.compile(
    r'(?<!\w)(?:' + '|'.join(re.escape(k) for k in sorted(_PHRASE_SLANG, key=len, reverse=True)) + r')(?!\w)',
    flags=re.IGNORECASE,
)

# Untuk kata tunggal, lindungi konteks tertentu:
# - 'dr.' (gelar), angka+'km' (satuan kilometer), dan 'u' di ekspresi matematika/assignment.
_WORD_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(k) for k in sorted(_WORD_SLANG, key=len, reverse=True)) + r')\b',
    flags=re.IGNORECASE,
)

_MATH_ASSIGN_BEFORE_RE = re.compile(r'[=+\-*/<>^(\[]\s*$')
_MATH_ASSIGN_AFTER_RE = re.compile(r'^\s*[=+\-*/<>^)\]]')
_NUMBER_BEFORE_RE = re.compile(r'\d\s*$')


def _replace_phrase(match: re.Match) -> str:
    return _PHRASE_SLANG[match.group(0).lower()]


def _replace_word(match: re.Match) -> str:
    raw = match.group(0)
    key = raw.lower()
    text = match.string
    start, end = match.span()

    # Lindungi gelar "Dr." dan satuan jarak "10 km"
    if key == 'dr' and end < len(text) and text[end] == '.':
        return raw
    if key == 'km' and _NUMBER_BEFORE_RE.search(text[:start]):
        return raw
    # Lindungi variabel huruf tunggal 'u' pada ekspresi matematika/kode (mis. u = 5, u + v)
    if key == 'u':
        if _MATH_ASSIGN_BEFORE_RE.search(text[:start]) or _MATH_ASSIGN_AFTER_RE.search(text[end:]):
            return raw

    return _WORD_SLANG[key]


def _normalize_prose(segment: str) -> str:
    text = segment.replace('’', "'").replace('`', "'")
    text = _PHRASE_RE.sub(_replace_phrase, text)
    text = _WORD_RE.sub(_replace_word, text)
    return text


def normalize_slang(text: str) -> str:
    """Expand Indonesian and English chat slang/abbreviations outside code blocks and URLs."""
    if not text:
        return ''
    normalized = unicodedata.normalize('NFKC', text)
    parts = _PROTECTED_SEGMENT_RE.split(normalized)
    out = []
    for idx, part in enumerate(parts):
        if not part:
            continue
        # Indeks ganjil adalah segmen yang cocok _PROTECTED_SEGMENT_RE (code/URL).
        if idx % 2 == 1:
            out.append(part)
        else:
            out.append(_normalize_prose(part))
    return ''.join(out)


def has_slang(text: str) -> bool:
    """Return True if normalize_slang expanded at least one slang token or phrase."""
    if not text or not text.strip():
        return False
    orig = re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().casefold()
    norm = re.sub(r'\s+', ' ', normalize_slang(text)).strip().casefold()
    return orig != norm
