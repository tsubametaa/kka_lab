import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()

parser = argparse.ArgumentParser(description='Unduh bobot model Hugging Face dan kunci revisinya di model_record.json.')
parser.add_argument('--model', default='', help='ID model Hugging Face (mis. Qwen/Qwen2.5-3B-Instruct atau Qwen/Qwen3-4B-Instruct-2507).')
parser.add_argument('--record', default='model_record.json', help='Path file model_record.json tujuan.')
parser.add_argument('--force', action='store_true', help='Perbarui model_record.json meskipun model_id berbeda dari record sebelumnya.')
args = parser.parse_args()

model_id = (args.model or os.getenv('MODEL_ID') or 'Qwen/Qwen2.5-3B-Instruct').strip()
record_path = Path(args.record)

if record_path.exists():
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record.get('model_id') != model_id:
        if args.force or bool(args.model):
            record = {'model_id': model_id, 'revision': HfApi().model_info(model_id).sha}
        else:
            raise ValueError(
                f"MODEL_ID ({model_id}) berbeda dari {record_path} ({record.get('model_id')}). "
                "Gunakan --model <id> atau --force untuk memperbarui record."
            )
else:
    record = {'model_id': model_id, 'revision': HfApi().model_info(model_id).sha}

snapshot_download(
    repo_id=record['model_id'],
    revision=record['revision'],
    allow_patterns=[
        '*.json', '*.safetensors', '*.model', 'tokenizer*',
        'vocab*', 'merges*', 'LICENSE*', 'README.md',
    ],
)
record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('Model diunduh ke cache Hugging Face.')
print(f'Simpan {record_path} dan gunakan juga saat training.')
print(json.dumps(record, indent=2))
