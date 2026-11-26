"""Audit kualitas, grounding konteks, keseimbangan, dan risiko bias dataset LoRA & DPO."""

import argparse
from collections import Counter
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.features.language.detector import detect_language

DEFAULT_SPLITS = {
    'train': ROOT / 'data' / 'lora' / 'lora_train.jsonl',
    'val': ROOT / 'data' / 'lora' / 'lora_val.jsonl',
    'test': ROOT / 'data' / 'lora' / 'lora_test.jsonl',
}
DEFAULT_DPO = ROOT / 'data' / 'lora' / 'dpo_preferences.jsonl'

_TRUNCATED_CONTEXT_RE = re.compile(r'(?:\b\d+\.|:\s*$|[a-z]{2,}\s*$)', re.IGNORECASE)
_PROPER_TECH_ENTITY_RE = re.compile(
    r'\b(?:BMKG|Waze|Google Maps|TensorFlow|PyTorch|Kubernetes|MongoDB|Redis|Figma|Jira|A\*|Dijkstra|LSTM|BERT|GPT-\d|Gemini \d)\b'
)
_BIASED_PHRASE_RE = re.compile(
    r'\b(?:laki-laki lebih (?:pintar|cocok|kuat)|perempuan (?:kurang cocok|tidak cocok)|'
    r'anak desa tidak bisa|hanya untuk anak (?:jenius|rpl)|pasti gagal)\b',
    re.IGNORECASE,
)
_PKTO_TUGAS_RE = re.compile(r'\bPKTO\b[^\n]{0,140}\bTugas\b', re.IGNORECASE)
_PKTO_TUJUAN_RE = re.compile(r'\bPKTO\b[^\n]{0,140}\bTujuan\b', re.IGNORECASE)


def _has_pkto_context_contradiction(ctx: str, asst: str) -> bool:
    """Deteksi jika jawaban asisten mengubah definisi PKTO (Tugas vs Tujuan) bertentangan dengan konteks."""
    if not ctx:
        return False
    ctx_tugas = bool(_PKTO_TUGAS_RE.search(ctx))
    ctx_tujuan = bool(_PKTO_TUJUAN_RE.search(ctx))
    asst_tugas = bool(_PKTO_TUGAS_RE.search(asst))
    asst_tujuan = bool(_PKTO_TUJUAN_RE.search(asst))
    return (ctx_tugas and asst_tujuan and not asst_tugas) or (ctx_tujuan and asst_tugas and not asst_tujuan)


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def audit_sft_split(rows):
    """Audit satu split SFT untuk keseimbangan, konteks terpotong, risiko ekstrasi luar konteks, dan konflik istilah."""
    lang_counts = Counter()
    mode_counts = Counter()
    sample_type_counts = Counter()
    truncated_context_ids = []
    ungrounded_entity_flags = []
    pktop_conflict_ids = []
    biased_target_ids = []
    terstruktur_no_q_lessons = []

    for row in rows:
        rid = row.get('id', '?')
        mode = row.get('mode', '?')
        stype = row.get('sample_type', 'lesson')
        ctx = (row.get('context') or '').strip()
        msgs = row.get('messages') or []
        last_user = next((m.get('content', '') for m in reversed(msgs) if m.get('role') == 'user'), '')
        last_asst = next((m.get('content', '') for m in reversed(msgs) if m.get('role') == 'assistant'), '')

        lang = detect_language(last_user)
        lang_counts[lang] += 1
        mode_counts[mode] += 1
        sample_type_counts[stype] += 1

        if ctx and len(ctx) >= 40 and ctx[-1] not in '.!?"\')`':
            truncated_context_ids.append(rid)
        elif ctx and re.search(r'(?:\b\d+\.|:\s*)$', ctx):
            truncated_context_ids.append(rid)

        # Cek entitas teknis spesifik yang muncul di jawaban asisten tetapi absen dari context & pertanyaan user.
        if ctx and stype != 'refusal':
            source_blob = f'{ctx} {last_user}'.casefold()
            leaked_entities = sorted({
                ent for ent in _PROPER_TECH_ENTITY_RE.findall(last_asst)
                if ent.casefold() not in source_blob
            })
            if leaked_entities:
                ungrounded_entity_flags.append({'id': rid, 'entities_not_in_context': leaked_entities})

        if _has_pkto_context_contradiction(ctx, last_asst):
            pktop_conflict_ids.append(rid)

        if _BIASED_PHRASE_RE.search(last_asst):
            biased_target_ids.append(rid)

        if mode == 'terstruktur' and stype == 'lesson' and '?' not in last_asst:
            terstruktur_no_q_lessons.append(rid)

    return {
        'rows': len(rows),
        'unique_groups': len({r.get('group_id') for r in rows}),
        'languages': dict(lang_counts),
        'modes': dict(mode_counts),
        'sample_types': dict(sample_type_counts),
        'truncated_context_count': len(truncated_context_ids),
        'truncated_context_sample': truncated_context_ids[:10],
        'ungrounded_entity_count': len(ungrounded_entity_flags),
        'ungrounded_entity_sample': ungrounded_entity_flags[:10],
        'pktop_terminology_conflicts': pktop_conflict_ids,
        'biased_phrase_hits': biased_target_ids,
        'terstruktur_lesson_without_question': terstruktur_no_q_lessons[:10],
    }

def audit_dpo_dataset(path: Path):
    rows = load_jsonl(path)
    if not rows:
        return {'exists': False, 'rows': 0}
    cats = Counter(r.get('category', 'unknown') for r in rows)
    langs = Counter(r.get('lang', 'unknown') for r in rows)
    modes = Counter(r.get('mode', 'unknown') for r in rows)
    return {
        'exists': True,
        'rows': len(rows),
        'unique_groups': len({r.get('group_id') for r in rows}),
        'categories': dict(cats),
        'languages': dict(langs),
        'modes': dict(modes),
    }

def run_full_audit(splits=None, dpo_path=DEFAULT_DPO):
    splits = splits or DEFAULT_SPLITS
    loaded = {name: load_jsonl(path) for name, path in splits.items()}
    group_sets = {name: {r.get('group_id') for r in rows} for name, rows in loaded.items()}

    leakage = {}
    names = list(loaded.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = sorted(group_sets[a] & group_sets[b])
            if overlap:
                leakage[f'{a}_vs_{b}'] = overlap

    report = {
        'splits': {name: audit_sft_split(rows) for name, rows in loaded.items()},
        'cross_split_group_leakage': leakage,
        'dpo': audit_dpo_dataset(dpo_path),
    }
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description='Audit kualitas dataset SFT LoRA & DPO KKA.')
    parser.add_argument('--save-report', default='', help='Opsional: simpan laporan JSON ke path ini.')
    parser.add_argument('--strict', action='store_true',
                        help='Keluar dengan kode 1 jika ada kebocoran split, konflik istilah, atau frasa bias.')
    args = parser.parse_args(argv)

    report = run_full_audit()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)

    if args.save_report:
        out = Path(args.save_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + '\n', encoding='utf-8')

    if args.strict:
        has_critical = bool(report['cross_split_group_leakage'])
        for split_info in report['splits'].values():
            if split_info['pktop_terminology_conflicts'] or split_info['biased_phrase_hits']:
                has_critical = True
        if has_critical:
            sys.exit(1)

if __name__ == '__main__':
    main()
