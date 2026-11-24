"""Interactive discussion topic suggestions when students are unsure what to discuss.

Saat siswa mengetik pesan seperti "saya mau diskusi, tapi bingung" atau "bingung mau
bahas apa", modul ini mendeteksi intent tersebut secara cepat dan menghasilkan daftar
rekomendasi terstruktur (`label` + `prompt`) yang bisa langsung diklik di frontend
untuk otomatis terkirim sebagai pesan chat berikutnya.
"""

import re
import unicodedata

from pydantic import BaseModel, Field, TypeAdapter

from app.features.language.slang import normalize_slang
from app.features.rag.auto_learner import match_kka_domain_terms

DEFAULT_SUGGESTION_COUNT = 4
MAX_SUGGESTION_COUNT = 6

# Sinyal bahwa siswa bingung memilih topik diskusi/belajar (ID & EN).
_CONFUSED_DISCUSSION_PATTERNS = (
    # "saya mau diskusi, tapi bingung", "aku pengen belajar tapi bingung mulai dari mana"
    re.compile(
        r'\b(?:diskusi|berdiskusi|ngobrol|bahas|membahas|belajar|tanya|bertanya|nanya|chat)\b'
        r'[^.\n]{0,40}\b(?:bingung|tidak\s+tahu|tak\s+tahu|gak\s+tau|ga\s+tau|nggak\s+tau|belum\s+tahu|buntu|blank)\b',
        flags=re.IGNORECASE,
    ),
    # "bingung mau diskusi apa", "bingung ingin bahas apa", "gak tau mau nanya apa"
    re.compile(
        r'\b(?:bingung|tidak\s+tahu|tak\s+tahu|gak\s+tau|ga\s+tau|nggak\s+tau|belum\s+tahu|buntu|blank)\b'
        r'[^.\n]{0,35}\b(?:mau|ingin|pengen|pingin|harus|mulai|tentang|soal)?\s*'
        r'(?:diskusi|berdiskusi|bahas|membahas|belajar|tanya|bertanya|nanya|ngobrol|topik|materi|dari\s+mana|apa)\b',
        flags=re.IGNORECASE,
    ),
    # "kasih rekomendasi topik diskusi", "saran topik dong", "enaknya bahas apa ya"
    re.compile(
        r'\b(?:kasih|beri(?:kan)?|minta|punya|ada)\b[^.\n]{0,25}'
        r'\b(?:rekomendasi|saran|ide|opsi|pilihan)\b[^.\n]{0,25}'
        r'\b(?:topik|materi|diskusi|bahasan|pertanyaan|obrolan|belajar)\b',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:rekomendasi|saran|ide|pilihan)\s+(?:topik|materi|diskusi|bahasan|pertanyaan)\b',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:enaknya|bagusnya|baiknya|kira-kira)\s+(?:kita\s+)?'
        r'(?:diskusi|berdiskusi|bahas|membahas|belajar|ngobrol)\s+apa\b',
        flags=re.IGNORECASE,
    ),
    # English variants
    re.compile(
        r'\b(?:want\s+to|wanna|like\s+to)\s+(?:discuss|chat|talk|learn|study)\b'
        r'[^.\n]{0,40}\b(?:confused|not\s+sure|unsure|don\'?t\s+know|no\s+idea)\b',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:confused|not\s+sure|unsure|don\'?t\s+know|no\s+idea)\b'
        r'[^.\n]{0,35}\b(?:what\s+to\s+(?:discuss|talk|ask|learn|study)|where\s+to\s+start|what\s+topic)\b',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'\b(?:suggest|recommend|give\s+me)\b[^.\n]{0,25}\b(?:topics?|ideas?|questions?)\b',
        flags=re.IGNORECASE,
    ),
    re.compile(
        r'\bwhat\s+should\s+(?:we|i)\s+(?:discuss|talk\s+about|learn|study)\b',
        flags=re.IGNORECASE,
    ),
)

# Istilah umum mata pelajaran yang masih wajar muncul saat siswa minta rekomendasi umum.
_GENERIC_SUBJECT_TERMS = frozenset({'kka', 'ai', 'koding', 'coding'})

# Katalog topik kurikulum KKA Kelas X (ID & EN).
_CURRICULUM_CATALOG = {
    'id': (
        {
            'id': 'kka-berpikir-komputasional',
            'topic': 'Berpikir Komputasional',
            'label': 'Berpikir Komputasional Sehari-hari',
            'prompt': 'Ayo kita diskusikan konsep berpikir komputasional (dekomposisi, pola, abstraksi, algoritma) dengan contoh kehidupan sehari-hari.',
            'keywords': ('komputasional', 'dekomposisi', 'abstraksi', 'pola'),
        },
        {
            'id': 'kka-algoritma-flowchart',
            'topic': 'Algoritma & Flowchart',
            'label': 'Menyusun Algoritma & Flowchart',
            'prompt': 'Jelaskan langkah-langkah menyusun algoritma dan flowchart dari masalah sederhana di sekolah.',
            'keywords': ('algoritma', 'flowchart', 'pseudocode'),
        },
        {
            'id': 'kka-pengantar-ai-ml',
            'topic': 'Pengantar AI & Machine Learning',
            'label': 'Cara Kerja AI & Machine Learning',
            'prompt': 'Bagaimana cara kerja kecerdasan artifisial dan apa bedanya dengan machine learning?',
            'keywords': ('machine learning', 'supervised', 'unsupervised', 'klasifikasi', 'regresi'),
        },
        {
            'id': 'kka-prompt-engineering',
            'topic': 'Prompt Engineering',
            'label': 'Teknik Prompt Engineering yang Efektif',
            'prompt': 'Bagaimana cara membuat prompt yang efektif agar jawaban AI akurat dan tidak berhalusinasi?',
            'keywords': ('prompt', 'prompting', 'halusinasi'),
        },
        {
            'id': 'kka-etika-privasi-ai',
            'topic': 'Etika & Privasi AI',
            'label': 'Etika, Bias, & Privasi Data pada AI',
            'prompt': 'Apa saja risiko bias dan masalah privasi data pada penggunaan AI, serta bagaimana cara mencegahnya?',
            'keywords': ('etika', 'privasi', 'bias', 'keamanan'),
        },
        {
            'id': 'kka-dasar-python',
            'topic': 'Dasar Pemrograman Python',
            'label': 'Dasar Variabel, Percabangan, & Perulangan',
            'prompt': 'Ayo kita bahas konsep dasar variabel, percabangan if-else, dan perulangan dalam koding dengan contoh sederhana.',
            'keywords': ('python', 'variabel', 'percabangan', 'perulangan', 'loop'),
        },
    ),
    'en': (
        {
            'id': 'kka-computational-thinking',
            'topic': 'Computational Thinking',
            'label': 'Computational Thinking in Daily Life',
            'prompt': 'Let us discuss the four pillars of computational thinking (decomposition, pattern recognition, abstraction, algorithms) with everyday examples.',
            'keywords': ('computational', 'decomposition', 'abstraction', 'pattern'),
        },
        {
            'id': 'kka-algorithms-flowcharts',
            'topic': 'Algorithms & Flowcharts',
            'label': 'Designing Algorithms & Flowcharts',
            'prompt': 'Explain step by step how to design an algorithm and flowchart for a simple real-world problem.',
            'keywords': ('algorithm', 'flowchart', 'pseudocode'),
        },
        {
            'id': 'kka-ai-machine-learning',
            'topic': 'AI & Machine Learning',
            'label': 'How AI & Machine Learning Work',
            'prompt': 'How does artificial intelligence work, and what is the difference between AI and machine learning?',
            'keywords': ('machine learning', 'supervised', 'unsupervised', 'classification'),
        },
        {
            'id': 'kka-prompt-engineering',
            'topic': 'Prompt Engineering',
            'label': 'Effective Prompt Engineering Tips',
            'prompt': 'How can we write effective prompts so AI gives accurate answers and avoids hallucinations?',
            'keywords': ('prompt', 'prompting', 'hallucination'),
        },
        {
            'id': 'kka-ai-ethics-privacy',
            'topic': 'AI Ethics & Privacy',
            'label': 'AI Ethics, Bias, & Data Privacy',
            'prompt': 'What are the main risks of bias and data privacy in AI systems, and how should students use AI responsibly?',
            'keywords': ('ethics', 'privacy', 'bias'),
        },
        {
            'id': 'kka-coding-basics',
            'topic': 'Coding Fundamentals',
            'label': 'Variables, Conditionals, & Loops',
            'prompt': 'Let us discuss variables, if-else conditionals, and loops in programming using a beginner-friendly example.',
            'keywords': ('python', 'variable', 'conditional', 'loop'),
        },
    ),
}


class DiscussionSuggestion(BaseModel):
    """Clickable discussion recommendation item returned to clients."""

    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=400)
    topic: str = Field(default='', max_length=120)


_SUGGESTIONS_ADAPTER = TypeAdapter(list[DiscussionSuggestion])


def _normalize(text: str) -> str:
    cleaned = unicodedata.normalize('NFKC', text or '')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def is_topic_suggestion_request(text: str) -> bool:
    """Return True when the student wants to discuss/learn but is unsure what topic to pick."""
    cleaned = _normalize(text)
    if not cleaned or len(cleaned) > 240:
        return False

    expanded = _normalize(normalize_slang(cleaned))

    # Jika siswa sudah menyebut istilah KKA spesifik (mis. "bingung soal algoritma/python/overfitting"),
    # berarti mereka butuh penjelasan materi spesifik, bukan menu pilih topik umum.
    matched_terms = match_kka_domain_terms(expanded)
    specific_terms = matched_terms - _GENERIC_SUBJECT_TERMS
    if specific_terms:
        return False

    return any(
        pattern.search(cleaned) or pattern.search(expanded)
        for pattern in _CONFUSED_DISCUSSION_PATTERNS
    )


def _doc_display_name(doc: dict) -> str:
    for key in ('topic', 'source_title', 'filename'):
        val = str(doc.get(key) or '').strip()
        if val:
            if key == 'filename' and '.' in val:
                val = val.rsplit('.', 1)[0]
            return re.sub(r'[_-]+', ' ', val).strip()[:70]
    return ''


def _build_user_doc_suggestions(user_docs, lang: str, max_items: int = 2) -> list[dict]:
    items = []
    seen_names = set()
    for doc in user_docs or ():
        if doc.get('status') != 'completed' or (doc.get('chunks_inserted') or 0) <= 0:
            continue
        name = _doc_display_name(doc)
        if not name or name.casefold() in seen_names:
            continue
        seen_names.add(name.casefold())
        doc_id = str(doc.get('doc_id') or len(items) + 1)[:32]
        if lang == 'en':
            label = f'Discuss My Material: {name}'[:120]
            prompt = f'Let us discuss the key concepts from my uploaded material "{name}".'
        else:
            label = f'Bahas Materiku: {name}'[:120]
            prompt = f'Ayo kita diskusikan konsep utama dari materi "{name}" yang sudah saya unggah.'
        items.append({
            'id': f'user-doc-{doc_id}',
            'label': label,
            'prompt': prompt,
            'topic': name,
        })
        if len(items) >= max_items:
            break
    return items


def build_discussion_suggestions(
    *,
    lang: str = 'id',
    history=None,
    user_docs=None,
    limit: int = DEFAULT_SUGGESTION_COUNT,
) -> list[dict]:
    """Build validated clickable discussion suggestions tailored to student docs & recent history."""
    safe_limit = max(1, min(int(limit), MAX_SUGGESTION_COUNT))
    target_lang = 'en' if lang == 'en' else 'id'
    catalog = _CURRICULUM_CATALOG[target_lang]

    recent_text = ' '.join(
        str(msg.get('content') or '') for msg in (history or ())[-6:]
    ).casefold()

    raw_items: list[dict] = []
    raw_items.extend(_build_user_doc_suggestions(user_docs, target_lang, max_items=min(2, safe_limit)))

    fresh = []
    seen_recent = []
    for entry in catalog:
        if any(kw in recent_text for kw in entry['keywords']):
            seen_recent.append(entry)
        else:
            fresh.append(entry)

    for entry in (*fresh, *seen_recent):
        if len(raw_items) >= safe_limit:
            break
        raw_items.append({
            'id': entry['id'],
            'label': entry['label'],
            'prompt': entry['prompt'],
            'topic': entry['topic'],
        })

    validated = _SUGGESTIONS_ADAPTER.validate_python(raw_items[:safe_limit])
    return [item.model_dump() for item in validated]


def build_suggestion_reply_context(suggestions: list[dict], lang: str = 'id') -> str:
    """Build synthetic system context so LLM introduces clickable recommendations naturally."""
    bullet_lines = '\n'.join(
        f'{idx}. {item["label"]}' for idx, item in enumerate(suggestions, start=1)
    )
    if lang == 'en':
        return (
            'System Note: The student wants to have a discussion but is unsure what topic to pick. '
            'The app UI is displaying clickable suggestion buttons right below your reply for these topics:\n'
            f'{bullet_lines}\n'
            'Respond warmly and concisely (2-3 short sentences): reassure them it is totally fine to feel unsure, '
            'briefly invite them to click any of the recommended topic buttons below to start right away (or type their own topic). '
            'Do not lecture on the topics yet, do not say material is missing, and do not tell them to ask the teacher.'
        )
    return (
        'Catatan Sistem: Siswa ingin berdiskusi namun sedang bingung memilih topik. '
        'Aplikasi sedang menampilkan tombol rekomendasi yang bisa langsung diklik tepat di bawah jawabanmu untuk topik berikut:\n'
        f'{bullet_lines}\n'
        'Balas dengan hangat dan singkat (2-3 kalimat): sampaikan tidak apa-apa jika masih bingung, lalu ajak siswa '
        'mengklik salah satu tombol rekomendasi topik di bawah untuk langsung mulai diskusi (atau mengetik topiknya sendiri). '
        'Jangan menjelaskan materi panjang lebar dulu, jangan menyatakan materi tidak ditemukan, dan jangan menyarankan bertanya kepada guru.'
    )
