import json
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()
model_id = os.getenv('MODEL_ID', 'Qwen/Qwen2.5-3B-Instruct')
record_path = Path('model_record.json')
if record_path.exists():
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record['model_id'] != model_id:
        raise ValueError('MODEL_ID berbeda dari record yang sudah ada.')
else:
    record = {'model_id': model_id, 'revision': HfApi().model_info(model_id).sha}
snapshot_download(
    repo_id=record['model_id'], revision=record['revision'],
    allow_patterns=[
        '*.json', '*.safetensors', '*.model', 'tokenizer*',
        'vocab*', 'merges*', 'LICENSE*', 'README.md',
    ],
)
record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
print('Model diunduh ke cache Hugging Face.')
print('Simpan model_record.json dan gunakan juga saat training.')
print(json.dumps(record, indent=2))
