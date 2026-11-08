"""Pre-download Baidu Unlimited-OCR weights and custom remote code to Hugging Face cache."""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()
model_id = os.getenv('OCR_MODEL_ID', 'baidu/Unlimited-OCR').strip() or 'baidu/Unlimited-OCR'
record_path = Path('ocr_model_record.json')

if record_path.exists():
    record = json.loads(record_path.read_text(encoding='utf-8'))
    if record['model_id'] != model_id:
        raise ValueError('OCR_MODEL_ID berbeda dari ocr_model_record.json yang sudah ada.')
else:
    record = {'model_id': model_id, 'revision': HfApi().model_info(model_id).sha}

snapshot_download(
    repo_id=record['model_id'],
    revision=record['revision'],
    allow_patterns=[
        '*.py', '*.json', '*.safetensors', '*.model', 'tokenizer*',
        'vocab*', 'merges*', 'preprocessor*', 'LICENSE*', 'README.md',
    ],
)
record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
print('Model Unlimited-OCR berhasil diunduh ke cache Hugging Face (tanpa perlu git clone).')
print(json.dumps(record, indent=2))

