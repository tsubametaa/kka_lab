"""Evaluation harness for AI KKA covering the 5 quality dimensions."""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES_PATH = ROOT / 'evals' / 'eval_cases.jsonl'
DATASET_FILES = [
    ROOT / 'data' / 'lora' / 'lora_train.jsonl',
    ROOT / 'data' / 'lora' / 'lora_val.jsonl',
    ROOT / 'data' / 'lora' / 'lora_test.jsonl',
]

_TAG_LEAK_RE = re.compile(r'(?<!\w)\[(?:S|M)\d+(?:\s*,\s*(?:S|M)\d+)*\]')
_SPECIAL_TOKEN_RE = re.compile(r'<\|[^|]+\|>|<｜[^｜]+｜>')
_METADATA_LEAK_RE = re.compile(
    r'(?:^|\n)\s*(?:Topik:|Sumber:|Konteks Sumber:|Memori Personal:)|\bauto_kka_[0-9a-f]+\b|\bdoc_id\b|\bskor\s*[:=]?\s*0\.\d+',
    re.IGNORECASE,
)
_STIFF_OPENER_RE = re.compile(
    r'^(?:pahami\s+bahwa|pelajari\s+bahwa|kuasai\s+|understand\s+that|learn\s+that|master\s+)',
    re.IGNORECASE,
)
_LIMITATION_MARKERS = (
    'tidak ditemukan',
    'tidak ada di materi',
    'tidak ada dalam materi',
    'tidak tercantum',
    'belum dibahas',
    'di luar materi',
    'tidak tersedia',
    'tidak membahas',
    'tanyakan kepada guru',
    'konfirmasi ke guru',
    'konfirmasi kepada guru',
    'diskusikan dengan guru',
    'bertanya kepada guru',
    'not found',
    'not covered',
    'not mentioned',
    'outside the material',
    'not in the material',
    'not included',
    'confirm with your teacher',
    'ask your teacher',
    'check with your teacher',
)

def load_jsonl(path):
    """Read non-empty JSONL lines from disk."""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_eval_cases(cases):
    """Validate structure and coverage of evaluation cases."""
    required = {
        'id', 'category', 'dimension', 'lang', 'mode', 'question',
        'context', 'history', 'required_points', 'forbidden_terms',
        'disallowed_history_terms', 'min_words', 'max_words',
        'expect_limitation', 'require_question', 'allow_question',
    }
    seen_ids = set()
    dimensions = set()
    for idx, case in enumerate(cases, start=1):
        missing = required - set(case.keys())
        if missing:
            raise ValueError(f'Case #{idx} missing keys: {sorted(missing)}')
        cid = case['id']
        if cid in seen_ids:
            raise ValueError(f'Duplicate case id: {cid}')
        seen_ids.add(cid)
        dimensions.add(case['dimension'])
    expected_dims = {'tuntas', 'akurat', 'proporsional', 'bersih', 'fokus'}
    if not expected_dims.issubset(dimensions):
        raise ValueError(f'Missing dimensions: {expected_dims - dimensions}')
    return {'total_cases': len(cases), 'dimensions': sorted(dimensions)}


def _has_artifact_leak(text):
    return bool(
        _TAG_LEAK_RE.search(text)
        or _SPECIAL_TOKEN_RE.search(text)
        or _METADATA_LEAK_RE.search(text)
    )


def _ends_cleanly(text):
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith('```'):
        return True
    return stripped[-1] in '.!?)"\''


def score_response(case, response):
    """Score one response across the 5 critique dimensions (0.0 to 1.0 each)."""
    text = (response or '').strip()
    lower = text.lower()
    words = text.split()
    word_count = len(words)
    issues = []

    # 1. Tuntas (Completeness)
    req_groups = case.get('required_points') or []
    if req_groups:
        matched = sum(1 for group in req_groups if any(term.lower() in lower for term in group))
        tuntas = matched / len(req_groups)
        if matched < len(req_groups):
            issues.append(f'tuntas:{matched}/{len(req_groups)} points')
    else:
        tuntas = 1.0 if word_count > 0 else 0.0

    # 2. Akurat (Accuracy / Grounding / Limitation honesty)
    akurat_checks = []
    forbidden_hits = [t for t in case.get('forbidden_terms', []) if t.lower() in lower]
    akurat_checks.append(0.0 if forbidden_hits else 1.0)
    if forbidden_hits:
        issues.append(f'akurat:forbidden={forbidden_hits}')
    if case.get('expect_limitation'):
        honest = any(marker in lower for marker in _LIMITATION_MARKERS)
        akurat_checks.append(1.0 if honest else 0.0)
        if not honest:
            issues.append('akurat:missing_limitation_or_teacher_note')
    akurat = sum(akurat_checks) / len(akurat_checks)

    # 3. Proporsional (Length proportionality + conditional trailing question)
    min_w = case.get('min_words', 1)
    max_w = case.get('max_words', 400)
    length_ok = min_w <= word_count <= max_w
    trailing_q = '?' in text.splitlines()[-1] if text.splitlines() else False
    q_policy_ok = True if case.get('allow_question', True) else not trailing_q
    proporsional = (0.6 if length_ok else 0.0) + (0.4 if q_policy_ok else 0.0)
    if not length_ok:
        issues.append(f'proporsional:words={word_count} not in [{min_w},{max_w}]')
    if not q_policy_ok:
        issues.append('proporsional:unexpected_trailing_question')

    # 4. Bersih (Artifact & style cleanliness)
    clean_checks = [
        0.0 if _has_artifact_leak(text) else 1.0,
        0.0 if _STIFF_OPENER_RE.search(text) else 1.0,
        1.0 if _ends_cleanly(text) else 0.0,
    ]
    bersih = sum(clean_checks) / len(clean_checks)
    if bersih < 1.0:
        issues.append('bersih:artifact_or_stiff_or_truncated')

    # 5. Fokus (Topic focus & guided question policy)
    fokus_checks = []
    topic_leaks = [t for t in case.get('disallowed_history_terms', []) if t.lower() in lower]
    fokus_checks.append(0.0 if topic_leaks else 1.0)
    if topic_leaks:
        issues.append(f'fokus:history_leak={topic_leaks}')
    if case.get('require_question'):
        has_q = '?' in text
        fokus_checks.append(1.0 if has_q else 0.0)
        if not has_q:
            issues.append('fokus:missing_guiding_question')
    fokus = sum(fokus_checks) / len(fokus_checks)

    overall = (tuntas + akurat + proporsional + bersih + fokus) / 5.0
    return {
        'id': case['id'],
        'dimension': case['dimension'],
        'scores': {
            'tuntas': round(tuntas, 3),
            'akurat': round(akurat, 3),
            'proporsional': round(proporsional, 3),
            'bersih': round(bersih, 3),
            'fokus': round(fokus, 3),
            'overall': round(overall, 3),
        },
        'passed': overall >= 0.999 and not issues,
        'issues': issues,
        'word_count': word_count,
    }


def summarize_results(results):
    """Aggregate average dimension scores and pass counts."""
    if not results:
        return {}
    dims = ('tuntas', 'akurat', 'proporsional', 'bersih', 'fokus', 'overall')
    averages = {
        d: round(sum(r['scores'][d] for r in results) / len(results), 4)
        for d in dims
    }
    passed = sum(1 for r in results if r['passed'])
    return {
        'total': len(results),
        'passed': passed,
        'pass_rate': round(passed / len(results), 4),
        'averages': averages,
    }


def audit_lora_datasets():
    """Inspect existing LoRA splits for artifact leakage and balance metrics."""
    report = {}
    for path in DATASET_FILES:
        if not path.exists():
            report[path.name] = {'exists': False}
            continue
        rows = load_jsonl(path)
        tag_leaks = 0
        meta_leaks = 0
        question_count = 0
        word_counts = []
        for row in rows:
            msgs = row.get('messages', [])
            assistant_text = ' '.join(
                m.get('content', '') for m in msgs if m.get('role') == 'assistant'
            )
            if _TAG_LEAK_RE.search(assistant_text):
                tag_leaks += 1
            if _METADATA_LEAK_RE.search(assistant_text):
                meta_leaks += 1
            if '?' in assistant_text:
                question_count += 1
            last_asst = next(
                (m.get('content', '') for m in reversed(msgs) if m.get('role') == 'assistant'),
                '',
            )
            word_counts.append(len(last_asst.split()))
        total = max(len(rows), 1)
        report[path.name] = {
            'rows': len(rows),
            'tag_leaks': tag_leaks,
            'metadata_leaks': meta_leaks,
            'has_question_pct': round(100.0 * question_count / total, 1),
            'min_words': min(word_counts) if word_counts else 0,
            'max_words': max(word_counts) if word_counts else 0,
            'over_160_words': sum(1 for w in word_counts if w > 160),
            'under_50_words': sum(1 for w in word_counts if w < 50),
        }
    return report


def _post_chat_json(endpoint, payload, api_key='', timeout=90):
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-API-Key'] = api_key
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def query_endpoint(endpoint, case, api_key='', timeout=90, user_id='eval-harness'):
    """Send an eval case (plus prior turns if present) to /api/chat and return answer text."""
    history_users = [m['content'] for m in case.get('history', []) if m.get('role') == 'user']
    session_id = None
    try:
        for turn_idx, prior_q in enumerate(history_users):
            resp = _post_chat_json(
                endpoint,
                {
                    'user_id': user_id,
                    'question': prior_q,
                    'session_id': session_id,
                    'mode': case['mode'],
                    'reset_context': turn_idx == 0,
                },
                api_key=api_key,
                timeout=timeout,
            )
            session_id = resp.get('session_id') or session_id
        body = _post_chat_json(
            endpoint,
            {
                'user_id': user_id,
                'question': case['question'],
                'session_id': session_id,
                'mode': case['mode'],
                'reset_context': not history_users,
            },
            api_key=api_key,
            timeout=timeout,
        )
        return body.get('answer', '')
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Endpoint request failed for {case["id"]}: {exc}') from exc

def run_offline_pipeline_check():
    """Verify scorer invariants and available sanitization/context helpers."""
    sample_case = {
        'id': 'SYN-CHECK-01',
        'dimension': 'bersih',
        'required_points': [['knn', 'jarak'], ['naive bayes', 'probabilitas']],
        'forbidden_terms': ['[s1]', 'topik:'],
        'disallowed_history_terms': ['flowchart'],
        'min_words': 15,
        'max_words': 120,
        'expect_limitation': False,
        'require_question': False,
        'allow_question': False,
    }
    clean = (
        'KNN memprediksi kelas berdasarkan jarak ke tetangga terdekat, '
        'sedangkan Naive Bayes menghitung probabilitas bersyarat tiap fitur. '
        'Indeks array Python seperti nilai[0] tetap aman dipakai saat contoh kode.'
    )
    dirty = '[S1] Topik: Klasifikasi\nPahami bahwa KNN memakai jarak. Mau lanjut?'
    r_clean = score_response(sample_case, clean)
    r_dirty = score_response(sample_case, dirty)
    if not r_clean['passed'] or r_dirty['passed']:
        raise AssertionError('Offline scorer check failed clean vs dirty separation.')
    return {'clean_score': r_clean['scores'], 'dirty_score': r_dirty['scores']}


def evaluate_local_adapter(cases, adapter_path='', save_report='', save_responses=''):
    """Jalankan evaluasi 40 kasus langsung lewat LocalAssistant menggunakan context & history di eval_cases.jsonl."""
    import os
    import time
    from pathlib import Path

    if adapter_path is not None:
        os.environ['MODEL_ADAPTER'] = adapter_path
    from app.core import load_record
    from app.features.chat.assistant import LocalAssistant

    record = load_record('model_record.json')
    asst = LocalAssistant()
    results = []
    resp_rows = []
    t0 = time.time()

    for idx, case in enumerate(cases, 1):
        cid = case['id']
        tc0 = time.time()
        ans = asst.answer(
            case['question'],
            case['mode'],
            case.get('history', []),
            case.get('context', ''),
            lang=case.get('lang', 'id'),
        )
        dt = round(time.time() - tc0, 2)
        res = score_response(case, ans)
        res['latency_sec'] = dt
        results.append(res)
        resp_rows.append({
            'id': cid,
            'dimension': res.get('dimension', case.get('dimension', '')),
            'latency_sec': dt,
            'answer': ans,
            'response': ans,
        })
        status = 'PASS' if res.get('passed') else f"FAIL({res.get('issues')})"
        print(f"[{idx:02d}/{len(cases)}] {cid} ({dt:.2f}s): {status} | scores={res.get('scores')}", flush=True)
        if save_responses:
            p_resp = Path(save_responses)
            p_resp.parent.mkdir(parents=True, exist_ok=True)
            p_resp.write_text(
                '\n'.join(json.dumps(r, ensure_ascii=False) for r in resp_rows) + '\n',
                encoding='utf-8',
            )

    total_sec = round(time.time() - t0, 2)
    summary = summarize_results(results)
    summary['total_eval_sec'] = total_sec
    summary['avg_latency_sec'] = round(total_sec / max(len(cases), 1), 2)
    summary['model_id'] = record.get('model_id', '')
    summary['adapter'] = os.environ.get('MODEL_ADAPTER', '')
    payload = {'summary': summary, 'results': results}
    if save_report:
        p_rep = Path(save_report)
        p_rep.parent.mkdir(parents=True, exist_ok=True)
        p_rep.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI KKA responses across 5 dimensions.')
    parser.add_argument('--cases', default=str(DEFAULT_CASES_PATH), help='Path to eval_cases.jsonl')
    parser.add_argument('--check-dataset', action='store_true', help='Validate eval cases & audit LoRA splits')
    parser.add_argument('--offline-pipeline-check', action='store_true', help='Run deterministic scorer checks')
    parser.add_argument('--responses', help='Path to JSONL file with {"id": ..., "response": ...}')
    parser.add_argument('--local-adapter', help='Path to LoRA adapter dir (e.g. adapters/qwen-kka-lora/final) to evaluate directly via LocalAssistant')
    parser.add_argument('--save-report', default='', help='Optional path to save JSON evaluation summary & results')
    parser.add_argument('--save-responses', default='', help='Optional path to save JSONL raw model answers')
    parser.add_argument('--endpoint', help='Live /api/chat endpoint URL to evaluate')
    parser.add_argument('--api-key', default='', help='Optional X-API-Key header for --endpoint')
    args = parser.parse_args()

    cases = load_jsonl(args.cases)
    case_summary = validate_eval_cases(cases)

    if args.check_dataset:
        output = {'eval_cases': case_summary, 'lora_audit': audit_lora_datasets()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.offline_pipeline_check:
        output = {'eval_cases': case_summary, 'offline_check': run_offline_pipeline_check()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.local_adapter is not None:
        payload = evaluate_local_adapter(
            cases,
            adapter_path=args.local_adapter,
            save_report=args.save_report,
            save_responses=args.save_responses,
        )
        print(json.dumps(payload['summary'], indent=2, ensure_ascii=False))
        return

    if args.responses:
        resp_map = {
            row['id']: (row.get('response') or row.get('answer', ''))
            for row in load_jsonl(args.responses)
        }
        results = [score_response(case, resp_map.get(case['id'], '')) for case in cases]
        payload = {'summary': summarize_results(results), 'results': results}
        if args.save_report:
            from pathlib import Path
            p_rep = Path(args.save_report)
            p_rep.parent.mkdir(parents=True, exist_ok=True)
            p_rep.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    if args.endpoint:
        results = []
        for case in cases:
            answer = query_endpoint(args.endpoint, case, api_key=args.api_key)
            results.append(score_response(case, answer))
        print(json.dumps({'summary': summarize_results(results), 'results': results}, indent=2, ensure_ascii=False))
        return

    parser.print_help()


if __name__ == '__main__':
    main()
