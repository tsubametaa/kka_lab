"""Stage-2 DPO (Direct Preference Optimization) alignment for KKA LoRA on RTX 5090 BF16.

Trains the model to strongly prefer grounded, unbiased, clean responses over
hallucinated, biased, prompt-injected, or leaky responses using TRL DPOTrainer.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import random
import re
import subprocess
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.core import load_record
from app.features.chat import make_system
from app.features.chat.prompts import POLICIES
from app.features.language import detect_language, format_user_message

_CHOSEN_LEAK_RE = re.compile(
    r'(?<!\w)\[(?:S|M)\d+(?:\s*[,;]\s*(?:S|M)\d+)*\]'
    r'|<\|[^|]+\|>|<｜[^｜]+｜>'
    r'|^\s*(?:Topik|Konteks Sumber|Memori Personal):'
    r'|\bauto_kka_[0-9a-f]{6,}\b',
    flags=re.IGNORECASE | re.MULTILINE,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Stage-2 DPO alignment anti-halusinasi & anti-bias untuk KKA (RTX 5090 BF16).'
    )
    parser.add_argument('--data', default='data/lora/dpo_preferences.jsonl',
                        help='Dataset preferensi JSONL (prompt/chosen/rejected).')
    parser.add_argument('--val-data', default='',
                        help='Opsional: dataset validasi DPO terpisah.')
    parser.add_argument('--record', default='model_record.json',
                        help='File model_record.json base model Qwen.')
    parser.add_argument('--sft-adapter', default='',
                        help='Path adaptor LoRA hasil SFT tahap 1 (mis. adapters/qwen-kka-lora-v3-qwen3-4b/final). '
                             'Jika diisi, adaptor SFT dilanjutkan langsung di DPO sehingga menghasilkan 1 adaptor drop-in.')
    parser.add_argument('--out', required=True, help='Folder output adaptor DPO.')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validasi argumen dan dataset DPO lalu keluar tanpa memuat GPU.')
    parser.add_argument('--precision', choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-6,
                        help='Learning rate DPO untuk LoRA (rekomendasi TRL: 5e-6).')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Koefisien KL penalty DPO (semakin besar = semakin dekat ke referensi).')
    parser.add_argument('--loss-type', choices=['sigmoid', 'robust', 'ipo'], default='robust',
                        help='Tipe loss DPO (robust + label smoothing membantu debias noise preferensi).')
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--max-length', type=int, default=1536,
                        help='Batas panjang gabungan prompt+completion (DPOConfig.max_length).')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--use-rslora', action='store_true')
    parser.add_argument('--target-modules', default='all-linear')
    parser.add_argument('--optim', default='adamw_torch_fused',
                        choices=['adamw_torch_fused', 'adamw_torch', 'paged_adamw_8bit'])
    parser.add_argument('--export-merged', default='',
                        help='Opsional: simpan model BF16 hasil merge_and_unload() untuk vLLM.')
    return parser


def validate_dpo_rows(rows):
    """Validasi skema, kebersihan jawaban chosen, dan pasangan chosen != rejected."""
    if not rows:
        raise ValueError('Dataset DPO tidak boleh kosong.')
    seen = set()
    for idx, row in enumerate(rows, start=1):
        for key in ('id', 'group_id', 'category', 'mode', 'context', 'question', 'chosen', 'rejected'):
            if key not in row:
                raise ValueError(f'Baris DPO #{idx} kekurangan kolom wajib: {key}')
        rid = str(row['id']).strip()
        if not rid or rid in seen:
            raise ValueError(f'ID DPO kosong atau ganda: {rid}')
        seen.add(rid)
        if row['mode'] not in POLICIES:
            raise ValueError(f'Mode tidak valid pada {rid}: {row["mode"]}')
        question = str(row['question']).strip()
        chosen = str(row['chosen']).strip()
        rejected = str(row['rejected']).strip()
        if not question or not chosen or not rejected:
            raise ValueError(f'question/chosen/rejected tidak boleh kosong pada {rid}')
        if chosen.casefold() == rejected.casefold():
            raise ValueError(f'chosen dan rejected identik pada {rid}')
        if _CHOSEN_LEAK_RE.search(chosen):
            raise ValueError(f'Jawaban chosen pada {rid} membocorkan artefak teknis/tag.')
        history = row.get('history') or []
        if len(history) % 2 != 0:
            raise ValueError(f'history pada {rid} harus berpasangan user/assistant.')


def split_dpo_groups(rows, seed=42, fraction=0.15):
    """Split train/val berdasarkan group_id agar tidak ada kebocoran skenario."""
    groups = sorted({r['group_id'] for r in rows})
    if len(groups) < 4:
        raise ValueError('Minimal 4 group_id unik untuk split DPO train/val.')
    random.Random(seed).shuffle(groups)
    nval = max(1, round(len(groups) * fraction))
    val_ids = set(groups[:nval])
    train_rows = [r for r in rows if r['group_id'] not in val_ids]
    val_rows = [r for r in rows if r['group_id'] in val_ids]
    return train_rows, val_rows


def build_dpo_example(row):
    """Ubah satu baris JSONL menjadi format percakapan eksplisit TRL DPOTrainer."""
    history = row.get('history') or []
    question = row['question'].strip()
    lang = row.get('lang') or detect_language(question, history)
    prompt_messages = [
        {'role': 'system', 'content': make_system(row['mode'], row.get('context', ''), lang=lang)},
        *history,
        {'role': 'user', 'content': format_user_message(question, lang)},
    ]
    return {
        'prompt': prompt_messages,
        'chosen': [{'role': 'assistant', 'content': row['chosen'].strip()}],
        'rejected': [{'role': 'assistant', 'content': row['rejected'].strip()}],
    }


def load_and_validate_dpo_splits(args):
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError('--batch-size dan --grad-accum minimal 1.')
    if args.max_length < 512:
        raise ValueError('--max-length minimal 512.')
    if args.beta <= 0:
        raise ValueError('--beta harus > 0.')
    if not (0.0 <= args.label_smoothing < 0.5):
        raise ValueError('--label-smoothing harus di rentang [0.0, 0.5).')

    raw = Path(args.data).read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    validate_dpo_rows(rows)

    if args.val_data:
        val_raw = Path(args.val_data).read_bytes()
        val_rows = [json.loads(line) for line in val_raw.decode('utf-8').splitlines() if line.strip()]
        validate_dpo_rows(val_rows)
        overlap = {r['group_id'] for r in rows} & {r['group_id'] for r in val_rows}
        if overlap:
            raise ValueError(f'Kebocoran group_id antara train dan val DPO: {sorted(overlap)[:5]}')
        return raw, hashlib.sha256(val_raw).hexdigest(), rows, val_rows

    train_rows, val_rows = split_dpo_groups(rows)
    return raw, None, train_rows, val_rows


def require_dpo_dependencies():
    required = ('torch', 'transformers', 'peft', 'trl', 'datasets')
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        raise RuntimeError(
            f"Pustaka DPO belum terpasang ({', '.join(missing)}).\n"
            'Jalankan: python -m pip install -r requirements-training.txt'
        )
    import torch
    from datasets import Dataset as HFDataset
    from peft import LoraConfig, PeftModel, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    return {
        'torch': torch,
        'HFDataset': HFDataset,
        'LoraConfig': LoraConfig,
        'PeftModel': PeftModel,
        'TaskType': TaskType,
        'AutoModelForCausalLM': AutoModelForCausalLM,
        'AutoTokenizer': AutoTokenizer,
        'set_seed': set_seed,
        'DPOConfig': DPOConfig,
        'DPOTrainer': DPOTrainer,
    }


def _filter_by_max_length(examples, tokenizer, max_length):
    """Pre-filter baris DPO yang melebihi max_length (TRL modern menghapus max_prompt_length)."""
    kept = []
    skipped = 0
    for ex in examples:
        chosen_len = len(tokenizer.apply_chat_template(
            ex['prompt'] + ex['chosen'], tokenize=True, add_generation_prompt=False
        ))
        rejected_len = len(tokenizer.apply_chat_template(
            ex['prompt'] + ex['rejected'], tokenize=True, add_generation_prompt=False
        ))
        if max(chosen_len, rejected_len) <= max_length:
            kept.append(ex)
        else:
            skipped += 1
    if not kept:
        raise ValueError(f'Semua contoh DPO melebihi max_length={max_length}. Naikkan --max-length.')
    if skipped:
        print(f'[INFO] Melewati {skipped} contoh DPO yang melebihi max_length={max_length}.')
    return kept


def main(argv=None):
    args = build_parser().parse_args(argv)
    raw, val_sha, train_rows, val_rows = load_and_validate_dpo_splits(args)
    record = load_record(args.record)

    if args.sft_adapter:
        sft_record_path = Path(args.sft_adapter) / 'model_record.json'
        if sft_record_path.exists():
            sft_record = load_record(str(sft_record_path))
            if sft_record != record:
                raise ValueError('model_record.json adaptor SFT berbeda dari --record saat ini.')

    if args.validate_only:
        cats = sorted({r['category'] for r in train_rows + val_rows})
        print(
            f'Validasi DPO OK: train={len(train_rows)}, val={len(val_rows)}, '
            f'categories={cats}, model={record.get("model_id")}, '
            f'sft_adapter={args.sft_adapter or "(fresh)"}, '
            f'loss_type={args.loss_type}, beta={args.beta}, lr={args.lr}'
        )
        return

    deps = require_dpo_dependencies()
    torch = deps['torch']
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU belum terdeteksi. Jalankan train_dpo.py di RTX 5090 / GPU CUDA.')

    use_bf16 = args.precision == 'bf16'
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU tidak mendukung bf16. Pakai --precision fp16.')
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if use_bf16:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    deps['set_seed'](42)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = deps['AutoTokenizer'].from_pretrained(
        record['model_id'], revision=record['revision']
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    train_examples = _filter_by_max_length(
        [build_dpo_example(r) for r in train_rows], tokenizer, args.max_length
    )
    val_examples = _filter_by_max_length(
        [build_dpo_example(r) for r in val_rows], tokenizer, args.max_length
    )
    train_ds = deps['HFDataset'].from_list(train_examples)
    val_ds = deps['HFDataset'].from_list(val_examples)

    base_model = deps['AutoModelForCausalLM'].from_pretrained(
        record['model_id'],
        revision=record['revision'],
        device_map={'': 0},
        dtype=compute_dtype,
        attn_implementation='sdpa',
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if hasattr(base_model, 'enable_input_require_grads'):
        base_model.enable_input_require_grads()

    peft_config = None
    if args.sft_adapter:
        # Lanjutkan langsung adaptor SFT agar hasil akhir tetap 1 adaptor LoRA drop-in;
        # TRL otomatis memakai base_model (adapter dinonaktifkan) sebagai KL reference tanpa copy kedua di VRAM.
        model = deps['PeftModel'].from_pretrained(
            base_model, args.sft_adapter, is_trainable=True
        )
        model.print_trainable_parameters()
    else:
        model = base_model
        target_modules = (
            'all-linear'
            if args.target_modules.strip() == 'all-linear'
            else [m.strip() for m in args.target_modules.split(',') if m.strip()]
        )
        peft_config = deps['LoraConfig'](
            task_type=deps['TaskType'].CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            use_rslora=args.use_rslora,
            bias='none',
        )

    dpo_cfg_kwargs = {
        'output_dir': str(out / 'checkpoints'),
        'num_train_epochs': args.epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': 2,
        'gradient_accumulation_steps': args.grad_accum,
        'learning_rate': args.lr,
        'beta': args.beta,
        'loss_type': args.loss_type,
        'label_smoothing': args.label_smoothing if args.loss_type in ('sigmoid', 'robust') else 0.0,
        'max_length': args.max_length,
        'warmup_ratio': 0.1,
        'lr_scheduler_type': 'cosine',
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'bf16': use_bf16,
        'fp16': not use_bf16,
        'tf32': use_bf16,
        'gradient_checkpointing': True,
        'gradient_checkpointing_kwargs': {'use_reentrant': False},
        'optim': args.optim,
        'logging_steps': 2,
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'save_total_limit': 2,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        'report_to': 'none',
        'seed': 42,
    }
    valid_cfg_params = set(inspect.signature(deps['DPOConfig'].__init__).parameters)
    filtered_cfg = {k: v for k, v in dpo_cfg_kwargs.items() if k in valid_cfg_params}
    dpo_config = deps['DPOConfig'](**filtered_cfg)

    trainer_kwargs = {
        'model': model,
        'ref_model': None,
        'args': dpo_config,
        'train_dataset': train_ds,
        'eval_dataset': val_ds,
        'peft_config': peft_config,
    }
    trainer_params = set(inspect.signature(deps['DPOTrainer'].__init__).parameters)
    if 'processing_class' in trainer_params:
        trainer_kwargs['processing_class'] = tokenizer
    elif 'tokenizer' in trainer_params:
        trainer_kwargs['tokenizer'] = tokenizer

    manifest = {
        'stage': 'dpo_alignment',
        'model': record,
        'sft_adapter': args.sft_adapter,
        'data_sha256': hashlib.sha256(raw).hexdigest(),
        'val_data_sha256': val_sha,
        'train_rows': len(train_examples),
        'validation_rows': len(val_examples),
        'categories': sorted({r['category'] for r in train_rows + val_rows}),
        'dpo_config': dpo_config.to_dict(),
        'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__,
    }
    (out / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str), encoding='utf-8'
    )
    (out / 'packages.lock.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        encoding='utf-8',
    )

    trainer = deps['DPOTrainer'](**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()

    final = out / 'final'
    trainer.model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (final / 'model_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (out / 'evaluation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    trainer.state.save_to_json(str(out / 'trainer_state.json'))
    print('Adaptor DPO tersimpan:', final)

    if args.export_merged:
        merged_dir = Path(args.export_merged)
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        (merged_dir / 'model_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print('Model merged BF16 (SFT+DPO) tersimpan:', merged_dir)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
