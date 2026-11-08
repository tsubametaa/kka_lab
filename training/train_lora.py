import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments, set_seed,
)
from peft import (
    LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training,
)
import sys
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from app.core import load_record
from app.features.chat import make_system
from app.features.dataset import encode_target, split_groups, validate_rows
from app.features.language import detect_language, format_user_message

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--record', default='model_record.json')
parser.add_argument('--out', required=True)
parser.add_argument('--resume', default='')
parser.add_argument('--allow-demo', action='store_true')
# Default di bawah sama dengan konfigurasi awal (Colab T4 / RTX 3050).
# GPU Ampere ke atas (RTX 30xx/40xx/50xx, A100, H100) dapat memakai --precision bf16.
parser.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--grad-accum', type=int, default=8)
args = parser.parse_args()
if args.batch_size < 1 or args.grad_accum < 1:
    raise ValueError('--batch-size dan --grad-accum minimal 1.')
if 'demo' in Path(args.data).name.lower() and not args.allow_demo:
    raise ValueError('Data demo bukan dataset penelitian. Pakai --allow-demo hanya untuk uji alur.')
if not torch.cuda.is_available():
    raise RuntimeError('Aktifkan GPU di Colab sebelum training.')
use_bf16 = args.precision == 'bf16'
if use_bf16 and not torch.cuda.is_bf16_supported():
    raise RuntimeError('GPU ini tidak mendukung bf16. Jalankan ulang dengan --precision fp16.')
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
set_seed(42)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
record = load_record(args.record)
raw = Path(args.data).read_bytes()
rows = [json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
validate_rows(rows)
train_rows, val_rows = split_groups(rows)
tokenizer = AutoTokenizer.from_pretrained(record['model_id'], revision=record['revision'])
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


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
                self.items.append(encode_target(tokenizer, messages, 1024))
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
        batch[key] = torch.tensor([
            f[key] + [pad] * (length - len(f[key])) for f in features
        ], dtype=torch.long)
    return batch


train_data, val_data = Dialogues(train_rows), Dialogues(val_rows)
quant = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    record['model_id'], revision=record['revision'],
    quantization_config=quant, device_map={'': 0}, dtype=compute_dtype,
    attn_implementation='sdpa',
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
    lora_dropout=0.05, target_modules=['q_proj', 'v_proj'], bias='none',
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
settings = TrainingArguments(
    output_dir=str(out / 'checkpoints'), num_train_epochs=2,
    per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
    gradient_accumulation_steps=args.grad_accum, learning_rate=2e-4,
    fp16=not use_bf16, bf16=use_bf16, gradient_checkpointing=True,
    optim='paged_adamw_8bit', logging_steps=5,
    eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model='eval_loss',
    greater_is_better=False, report_to='none', seed=42,
    dataloader_num_workers=0, remove_unused_columns=False,
    label_names=['labels'],
)
manifest = {
    'model': record, 'data_sha256': hashlib.sha256(raw).hexdigest(),
    'train_rows': len(train_rows), 'validation_rows': len(val_rows),
    'train_groups': sorted({r['group_id'] for r in train_rows}),
    'validation_groups': sorted({r['group_id'] for r in val_rows}),
    'pedagogy_requirements': [
        'at_least_two_steps', 'reasoning_explanation', 'critical_thinking_prompt',
    ],
    'max_length': 1024, 'lora': config.to_dict(),
    'torch': torch.__version__, 'cuda': torch.version.cuda,
    'gpu': torch.cuda.get_device_name(0),
    'training_args': settings.to_dict(),
}
(out / 'run_manifest.json').write_text(
    json.dumps(manifest, indent=2, default=str), encoding='utf-8'
)
(out / 'packages.lock.txt').write_text(
    subprocess.check_output(['python', '-m', 'pip', 'freeze'], text=True),
    encoding='utf-8',
)
trainer = Trainer(
    model=model, args=settings, train_dataset=train_data,
    eval_dataset=val_data, data_collator=collate,
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
