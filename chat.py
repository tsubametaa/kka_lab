import argparse
import json
import time
import warnings
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from app.core import load_record
from app.features.chat import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_SAMPLING_CONFIG,
    build_generation_kwargs,
    fit_messages_to_budget,
    make_system,
    resolve_max_prompt_tokens,
)
from app.features.language import (
    bilingual_queries,
    detect_language,
    ensure_response_language,
    format_user_message,
)
from app.features.persona import PersonaError, clean_persona
from app.features.rag import (
    RagUnavailableError,
    build_context,
    resolve_query_topic,
    retrieve,
)

warnings.filterwarnings('ignore', message=r'.*_check_is_size.*', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['umum', 'terstruktur'], default='terstruktur')
parser.add_argument('--adapter', default='')
parser.add_argument('--record', default='model_record.json')
parser.add_argument('--no-rag', action='store_true')
parser.add_argument('--log', default='')
parser.add_argument('--sources', default='', help='JSON hasil rag.py --save untuk konteks tetap')
parser.add_argument('--rag-per-sesi', action='store_true',
                    help='Ambil sumber sekali di awal sesi saja (protokol awal tutorial)')
parser.add_argument('--max-prompt-tokens', type=int, default=0,
                    help='Batas maksimum token prompt (default: 0 = otomatis maksimum model ~32K token)')
parser.add_argument('--max-new-tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS)
parser.add_argument('--greedy', action='store_true',
                    help='Gunakan greedy decoding (do_sample=False) untuk pengujian deterministik')
parser.add_argument('--temperature', type=float, default=DEFAULT_SAMPLING_CONFIG['temperature'])
parser.add_argument('--top-p', type=float, default=DEFAULT_SAMPLING_CONFIG['top_p'])
parser.add_argument('--min-p', type=float, default=DEFAULT_SAMPLING_CONFIG['min_p'])
parser.add_argument('--repetition-penalty', type=float, default=DEFAULT_SAMPLING_CONFIG['repetition_penalty'])
parser.add_argument('--max-history', type=int, default=4,
                    help='Jumlah giliran tanya-jawab terakhir yang diingat')
parser.add_argument('--debug', action='store_true',
                    help='Tampilkan label sumber, skor similarity, query RAG, dan waktu')
parser.add_argument('--tanpa-sumber', action='store_true', help='Sembunyikan baris sumber')
parser.add_argument('--persona', default='',
                    help='Gaya jawaban pilihan siswa, mis. "tsundere, bilang Hmph kalau tidak setuju"')
parser.add_argument('--session', default='',
                    help='ID sesi percakapan untuk melanjutkan chat di Astra DB (jika kosong, buat sesi baru)')
parser.add_argument('--user-id', default='siswa_01',
                    help='ID user untuk riwayat chat di Astra DB (default: siswa_01)')
parser.add_argument('--no-astra-history', action='store_true',
                    help='Jangan simpan riwayat chat ke Astra DB (hanya memori lokal sementara)')
args = parser.parse_args()
try:
    persona = clean_persona(args.persona)
except PersonaError as err:
    raise SystemExit(f'Persona ditolak: {err}')
if not torch.cuda.is_available():
    raise RuntimeError('CUDA belum terdeteksi. Periksa driver dan PyTorch GPU.')
record = load_record(args.record)
tokenizer = AutoTokenizer.from_pretrained(
    record['model_id'], revision=record['revision']
)
tokenizer.pad_token = tokenizer.eos_token
quant = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    record['model_id'], revision=record['revision'],
    quantization_config=quant, device_map={'': 0},
    dtype=torch.float16, attn_implementation='sdpa',
)
if args.adapter:
    other = load_record(Path(args.adapter) / 'model_record.json')
    if other != record:
        raise ValueError('Model/revisi adaptor berbeda dari model lokal.')
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
model.eval()
max_prompt_tokens = resolve_max_prompt_tokens(
    model, max_new_tokens=args.max_new_tokens, configured=args.max_prompt_tokens
)

fixed_docs = None
if not args.no_rag:
    from app.features.rag import retrieve, build_context, RagUnavailableError
    if args.sources:
        saved = json.loads(Path(args.sources).read_text(encoding='utf-8'))
        fixed_docs = saved['docs']
        print('Menggunakan sumber tetap untuk skenario:', saved['question'])

if args.no_rag:
    rag_strategy = 'nonaktif'
elif fixed_docs is not None:
    rag_strategy = 'sumber_tetap'
elif args.rag_per_sesi:
    rag_strategy = 'per_sesi'
else:
    rag_strategy = 'per_pertanyaan'


def format_similarity(value):
    return f'{value:.3f}' if isinstance(value, (int, float)) else '-'


history, context, sources, topic = [], '', [], ''
session_rag_loaded = False

# Inisialisasi penyimpanan riwayat obrolan di Astra DB
astra_store = None
session_id = args.session.strip() or None
if not args.no_astra_history:
    try:
        from app.features.memory.astra_store import AstraChatStore
        astra_store = AstraChatStore()
        if session_id:
            loaded_history = astra_store.history(session_id, limit=args.max_history)
            if loaded_history:
                history.extend(loaded_history)
                print(f'Melanjutkan sesi Astra DB: {session_id} ({len(loaded_history)//2} tanya-jawab dimuat)')
            else:
                print(f'Sesi Astra DB [{session_id}] belum memiliki riwayat obrolan.')
        else:
            session_id = astra_store.create_session(args.user_id)
            print(f'Sesi chat Astra DB aktif: {session_id}')
    except Exception as exc:
        if args.debug:
            print(f'[debug] Astra DB chat history tidak aktif: {exc}')

print('Siap. /baru = sesi baru; /keluar = berhenti.')
print('/persona <gaya> = atur gaya jawaban; /persona = lihat; /persona hapus = kembali normal.')
if persona:
    print('Persona aktif:', persona)
if rag_strategy == 'per_pertanyaan':
    print('RAG mengambil sumber baru di setiap pertanyaan.')
elif rag_strategy == 'per_sesi':
    print('RAG mengambil sumber saat awal sesi; /baru untuk mengganti topik.')
while True:
    question = input('\nAnda: ').strip()
    if question == '/keluar':
        break
    if question == '/baru':
        history, context, sources, topic = [], '', [], ''
        session_rag_loaded = False
        if astra_store:
            session_id = astra_store.create_session(args.user_id)
            print(f'Sesi baru dimulai (ID Sesi Astra DB: {session_id}).')
        else:
            print('Sesi baru dimulai.')
        continue
    if question == '/persona' or question.startswith('/persona '):
        value = question[len('/persona'):].strip()
        if not value:
            print('Persona aktif:', persona or '(tidak ada)')
        elif value.casefold() in ('hapus', 'reset', 'off'):
            persona = ''
            print('Persona dihapus; gaya jawaban kembali normal.')
        else:
            try:
                persona = clean_persona(value)
                print('Persona disimpan untuk sesi ini:', persona)
            except PersonaError as err:
                print('Persona ditolak:', err)
        continue
    if not question:
        continue
    query = None
    rag_error = None
    if rag_strategy == 'sumber_tetap':
        if not context:
            context, sources = build_context(fixed_docs, tokenizer)
    elif rag_strategy == 'per_pertanyaan' or (rag_strategy == 'per_sesi' and not session_rag_loaded):
        query, topic = resolve_query_topic(question, topic)
        search_queries = bilingual_queries(query, model, tokenizer)
        try:
            context, sources = build_context(retrieve(search_queries), tokenizer)
            session_rag_loaded = True
        except RagUnavailableError as err:
            rag_error = str(err)
            print(f'\n[peringatan] {rag_error}')
            print('[peringatan] Menjawab sementara tanpa konteks materi RAG.')
            context, sources = '', []
    recent = history[-2 * args.max_history:] if args.max_history > 0 else []
    lang = detect_language(question, recent)
    user_message = format_user_message(question, lang)
    started = time.perf_counter()
    budget = max_prompt_tokens
    oom_cls = getattr(torch.cuda, 'OutOfMemoryError', ())
    try:
        while True:
            messages, prompt_tokens = fit_messages_to_budget(
                args.mode,
                context,
                user_message,
                history=recent,
                tokenizer=tokenizer,
                max_prompt_tokens=budget,
                memory='',
                lang=lang,
                persona=persona,
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            input_len = inputs['input_ids'].shape[1]
            inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
            try:
                gen_kwargs = build_generation_kwargs(
                    tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=not args.greedy,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    min_p=args.min_p,
                    repetition_penalty=args.repetition_penalty,
                )
                with torch.inference_mode():
                    output = model.generate(**inputs, **gen_kwargs)
                break
            except RuntimeError as exc:
                del inputs
                torch.cuda.empty_cache()
                is_oom = (oom_cls and isinstance(exc, oom_cls)) or 'out of memory' in str(exc).lower()
                if not is_oom or prompt_tokens <= 1536:
                    raise
                budget = max(1536, min(budget, prompt_tokens) // 2)
                print(f'\n[peringatan] Memori GPU penuh; menyesuaikan batas konteks ke {budget} token...')
    except ValueError as err:
        print(f'\n[peringatan] {err}')
        continue
    new_ids = output[0, input_len:]
    answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    answer = ensure_response_language(
        answer, lang, model, tokenizer, max_new_tokens=args.max_new_tokens
    )
    seconds = round(time.perf_counter() - started, 3)
    print('\nAsisten:', answer)
    if args.debug:
        print('\n[debug] Sumber:', ', '.join(
            f'[{s["label"]}] {s["title"]} (skor {format_similarity(s.get("similarity"))})'
            for s in sources
        ) or 'tidak ada')
        print('[debug] Bahasa:', lang,
              '| Query RAG:', query or '(tidak mencari ulang)',
              '| Token prompt:', prompt_tokens,
              '| Token jawaban:', len(new_ids),
              '| Waktu generasi:', seconds, 'detik')
    elif sources and not args.tanpa_sumber:
        titles = list(dict.fromkeys(s['title'] for s in sources))
        print('\n📚 Sumber:', '; '.join(titles))
    history.extend([
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer},
    ])
    if astra_store and session_id:
        try:
            astra_store.save_exchange(
                user_id=args.user_id,
                session_id=session_id,
                question=question,
                answer=answer,
                mode=args.mode,
                sources=sources,
                training_consent=False,
            )
            if args.debug:
                print(f'[debug] Riwayat chat tersimpan di Astra DB (Sesi: {session_id})')
        except Exception as exc:
            if args.debug:
                print(f'[debug] Gagal menyimpan chat ke Astra DB: {exc}')
    if args.log:
        path = Path(args.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            'mode': args.mode, 'adapter': args.adapter or None,
            'model': record, 'question': question, 'answer': answer,
            'sources': sources, 'generation_seconds': seconds,
            'rag_strategy': rag_strategy, 'retrieval_query': query,
            'max_new_tokens': args.max_new_tokens, 'rag_error': rag_error,
            'persona': persona or None,
        }
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
