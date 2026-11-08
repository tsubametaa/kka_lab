import json
from pathlib import Path


def load_record(path='model_record.json'):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    if not record.get('model_id') or not record.get('revision'):
        raise ValueError('Record model belum lengkap. Jalankan download_model.py.')
    return record