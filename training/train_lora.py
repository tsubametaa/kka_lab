"""Fine-tune Qwen2.5-3B-Instruct LoRA adapter for KKA pedagogical mentoring."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.core import load_record
from app.features.chat import make_system
from app.features.dataset import encode_target, split_groups, validate_rows
from app.features.language import detect_language, format_user_message


def build_parser():
    parser = argparse.ArgumentParser(description='Train LoRA adapter KKA.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--val-data', default='')
    parser.add_argument('--record', default='model_record.json')
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', default='')
    parser.add_argument('--allow-demo', action='store_true')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validasi argumen dan dataset lalu keluar tanpa memuat GPU/model.')
    # Default cocok untuk T4/RTX 3050; GPU Ampere+ (RTX 30xx/40xx/50xx, A100, H100) pakai --precision bf16.
    parser.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--target-modules', default='all-linear')
    parser.add_argument('--early-stopping-patience', type=int, default=2)
    return parser

def load_and_validate_splits(args):
    if args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError('--batch-size dan --grad-accum minimal 1.')
    if args.max_length < 512:
        raise ValueError('--max-length minimal 512.')
    if 'demo' in Path(args.data).name.lower() and not args.allow_demo:
        raise ValueError('Data demo bukan dataset penelitian. Pakai --allow-demo hanya untuk uji alur.')

    raw = Path(args.data).read_bytes()
    rows = [json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
    validate_rows(rows)

    if args.val_data:
        val_raw = Path(args.val_data).read_bytes()
        val_rows = [json.loads(x) for x in val_raw.decode('utf-8').splitlines() if x.strip()]
        validate_rows(val_rows)
        train_rows = rows
        overlap = {r['group_id'] for r in train_rows} & {r['group_id'] for r in val_rows}
        if overlap:
            raise ValueError(f'Kebocoran group_id antara train dan val-data: {sorted(overlap)[:5]}')
        val_sha = hashlib.sha256(val_raw).hexdigest()
    else:
        train_rows, val_rows = split_groups(rows)
        val_sha = None
    return raw, val_sha, train_rows, val_rows


def require_training_dependencies():
    """Pastikan pustaka training GPU tersedia sebelum memuat model."""
    required = ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')
    missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
    if missing:
        raise RuntimeError(
            f"Pustaka training belum terpasang di interpreter Python ini ({sys.executable}): "
            f"{', '.join(missing)}.\n"
            'Solusi:\n'
            '1) Jika menjalankan di WSL (.venv): aktifkan `source .venv/bin/activate`, atau\n'
            '2) Pasang dependensi inti: `python -m pip install -r requirements-core.txt`, atau\n'
            '3) Jalankan training di instance Cloud GPU sesuai PANDUAN_CLOUD_GPU.md.'
        )

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    return {
        'torch': torch,
        'Dataset': Dataset,
        'AutoModelForCausalLM': AutoModelForCausalLM,
        'AutoTokenizer': AutoTokenizer,
        'BitsAndBytesConfig': BitsAndBytesConfig,
        'EarlyStoppingCallback': EarlyStoppingCallback,
        'Trainer': Trainer,
        'TrainingArguments': TrainingArguments,
        'set_seed': set_seed,
        'LoraConfig': LoraConfig,
        'TaskType': TaskType,
        'get_peft_model': get_peft_model,
        'prepare_model_for_kbit_training': prepare_model_for_kbit_training,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    raw, val_sha, train_rows, val_rows = load_and_validate_splits(args)
    record = load_record(args.record)

    if args.validate_only:
        print(
            f'Validasi OK: train={len(train_rows)} baris, val={len(val_rows)} baris, '
            f'model={record.get("model_id")}'
        )
        return

    deps = require_training_dependencies()
    torch = deps['torch']
    if not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA GPU belum tersedia di Python ini ({sys.executable}, torch={torch.__version__}). '
            'Jalankan training di GPU NVIDIA (WSL dengan CUDA atau Cloud GPU sesuai PANDUAN_CLOUD_GPU.md).'
        )

    use_bf16 = args.precision == 'bf16'
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU ini tidak mendukung bf16. Jalankan ulang dengan --precision fp16.')
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    deps['set_seed'](42)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = deps['AutoTokenizer'].from_pretrained(
        record['model_id'], revision=record['revision']
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    Dataset = deps['Dataset']

    class Dialogues(Dataset):
        def __init__(self, examples):
            self.items = []
            for row in examples:
                prior = row['messages'][:-2]
                last_user = row['messages'][-2]['content']
                lang = detect_language(last_user, prior)
                messages = [
                    {'role': 'system', 'content': make_system(row['mode'], row['context'], lang=lang)},
                    *prior,
                    {'role': 'user', 'content': format_user_message(last_user, lang)},
                    row['messages'][-1],
                ]
                try:
                    self.items.append(encode_target(tokenizer, messages, args.max_length))
                except ValueError as exc:
                    raise ValueError(f'{row["id"]}: {exc}') from exc

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate(features):
        length = max(len(x['input_ids']) for x in features)
        batch = {}
        pads = {'input_ids': tokenizer.pad_token_id, 'attention_mask': 0, 'labels': -100}
        for key, pad in pads.items():
            batch[key] = torch.tensor(
                [f[key] + [pad] * (length - len(f[key])) for f in features],
                dtype=torch.long,
            )
        return batch

    train_data, val_data = Dialogues(train_rows), Dialogues(val_rows)
    quant = deps['BitsAndBytesConfig'](
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = deps['AutoModelForCausalLM'].from_pretrained(
        record['model_id'],
        revision=record['revision'],
        quantization_config=quant,
        device_map={'': 0},
        dtype=compute_dtype,
        attn_implementation='sdpa',
    )
    model = deps['prepare_model_for_kbit_training'](model)
    model.config.use_cache = False

    target_modules = (
        'all-linear'
        if args.target_modules.strip() == 'all-linear'
        else [m.strip() for m in args.target_modules.split(',') if m.strip()]
    )
    config = deps['LoraConfig'](
        task_type=deps['TaskType'].CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias='none',
    )
    model = deps['get_peft_model'](model, config)
    model.print_trainable_parameters()

    settings = deps['TrainingArguments'](
        output_dir=str(out / 'checkpoints'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type='cosine',
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        optim='paged_adamw_8bit',
        logging_steps=5,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to='none',
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        label_names=['labels'],
    )

    manifest = {
        'model': record,
        'data_sha256': hashlib.sha256(raw).hexdigest(),
        'val_data_sha256': val_sha,
        'train_rows': len(train_rows),
        'validation_rows': len(val_rows),
        'train_groups': sorted({r['group_id'] for r in train_rows}),
        'validation_groups': sorted({r['group_id'] for r in val_rows}),
        'pedagogy_requirements': [
            'natural_conversational_flow',
            'no_stiff_template_opener',
            'reasoning_explanation',
            'concrete_smk_sma_case_simulation',
            'topic_shift_clean_focus',
            'critical_thinking_prompt',
        ],
        'max_length': args.max_length,
        'lora': config.to_dict(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'training_args': settings.to_dict(),
    }
    (out / 'run_manifest.json').write_text(
        json.dumps(manifest, indent=2, default=str), encoding='utf-8'
    )
    (out / 'packages.lock.txt').write_text(
        subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
        encoding='utf-8',
    )

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            deps['EarlyStoppingCallback'](
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.005,
            )
        )

    trainer = deps['Trainer'](
        model=model,
        args=settings,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    metrics = trainer.evaluate()

    final = out / 'final'
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (final / 'model_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    (out / 'evaluation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    trainer.state.save_to_json(str(out / 'trainer_state.json'))
    print('Adaptor tersimpan:', final)
    print('Nilai validation loss bukan bukti kenaikan pemahaman siswa.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
