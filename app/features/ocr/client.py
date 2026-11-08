"""Client wrapper for Baidu Unlimited-OCR (vLLM server, Transformers, or native PDF text)."""

import base64
import json
import mimetypes
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from app.features.ocr.exceptions import DocumentValidationError, OcrUnavailableError

DEFAULT_OCR_MODEL = 'baidu/Unlimited-OCR'
DEFAULT_OCR_BASE_URL = 'http://127.0.0.1:8001/v1'
DEFAULT_OCR_PROMPT = '<image>document parsing.'
TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}

_EOS_TOKENS_RE = re.compile(
    r'(<｜end▁of▁sentence｜>|<\|end_of_sentence\|>|<\|endoftext\|>|<\|im_end\|>)'
)
_OUTER_FENCE_RE = re.compile(r'^\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*$', re.DOTALL)

_transformers_lock = threading.Lock()
_transformers_bundle = None


@dataclass(frozen=True)
class PageMarkdown:
    page_number: int
    markdown: str

def clean_ocr_markdown(raw_text):
    text = str(raw_text or '')
    text = _EOS_TOKENS_RE.sub('', text)
    if text.lstrip().startswith('<image>'):
        text = text.lstrip()[len('<image>'):]
    fenced = _OUTER_FENCE_RE.match(text)
    if fenced and '```' not in fenced.group(1):
        text = fenced.group(1)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _encode_image_data_url(image_path):
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or 'image/png'
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{encoded}'

def _parse_page_via_vllm(image_path, *, base_url, model_id, api_key, prompt,
                         max_tokens, timeout, retries=2, backoff=1.5):
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    data_url = _encode_image_data_url(image_path)
    payload = json.dumps({
        'model': model_id,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': data_url}},
                {'type': 'text', 'text': prompt},
            ],
        }],
        'temperature': 0.0,
        'max_tokens': max_tokens,
    }).encode('utf-8')

    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    for attempt in range(retries + 1):
        req = urllib.request.Request(endpoint, data=payload, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode('utf-8'))
            choices = body.get('choices') or []
            if not choices:
                raise OcrUnavailableError('Respons Unlimited-OCR kosong (choices tidak ada).')
            content = (choices[0].get('message') or {}).get('content', '')
            return clean_ocr_markdown(content)
        except urllib.error.HTTPError as exc:
            if exc.code in TRANSIENT_HTTP_CODES and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            detail = exc.read().decode('utf-8', errors='replace')[:300]
            raise OcrUnavailableError(
                f'Unlimited-OCR mengembalikan HTTP {exc.code}: {detail}'
            ) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            raise OcrUnavailableError(
                f'Gagal menghubungi server Unlimited-OCR di {base_url} ({exc}). '
                'Pastikan proses OCR aktif (jalankan scripts/serve_ocr_vllm.sh).'
            ) from exc

def _load_transformers_ocr(model_id):
    global _transformers_bundle
    if _transformers_bundle is not None and _transformers_bundle[0] == model_id:
        return _transformers_bundle[1], _transformers_bundle[2]

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise OcrUnavailableError(
            "Pustaka 'torch' dan 'transformers' diperlukan untuk OCR_BACKEND=transformers."
        ) from exc

    if not torch.cuda.is_available():
        raise OcrUnavailableError('CUDA GPU diperlukan untuk menjalankan Unlimited-OCR.')

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    ).eval().cuda()
    _transformers_bundle = (model_id, tokenizer, model)
    return tokenizer, model

def _parse_via_transformers(pages, *, model_id, prompt, work_dir):
    out_dir = Path(work_dir or '.') / 'ocr_out'
    out_dir.mkdir(parents=True, exist_ok=True)

    with _transformers_lock:
        tokenizer, model = _load_transformers_ocr(model_id)
        results = []
        for page in pages:
            page_dir = out_dir / f'p_{page.page_number:04d}'
            page_dir.mkdir(parents=True, exist_ok=True)
            raw = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=str(page.image_path),
                output_path=str(page_dir),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                eval_mode=True,
            )
            if not raw:
                saved_files = sorted(page_dir.glob('*.md')) + sorted(page_dir.glob('*.mmd')) + sorted(page_dir.glob('*.txt'))
                if saved_files:
                    raw = saved_files[0].read_text(encoding='utf-8', errors='replace')
            cleaned = clean_ocr_markdown(raw or '')
            if cleaned:
                results.append(PageMarkdown(page_number=page.page_number, markdown=cleaned))
        return results

def parse_pages_to_markdown(pages, *, backend=None, work_dir=None):
    """Konversi daftar RenderedPage menjadi daftar PageMarkdown bersih."""
    if not pages:
        return []

    mode = (backend or os.getenv('OCR_BACKEND', 'vllm')).strip().lower()
    if mode not in ('vllm', 'transformers', 'native_pdf'):
        raise ValueError(
            f"OCR_BACKEND '{mode}' tidak dikenal. Pilih: vllm, transformers, atau native_pdf."
        )

    if mode == 'native_pdf':
        extracted = [
            PageMarkdown(page_number=p.page_number, markdown=clean_ocr_markdown(p.native_text))
            for p in pages
            if (p.native_text or '').strip()
        ]
        if not extracted:
            raise DocumentValidationError(
                'Dokumen ini berupa gambar/hasil scan tanpa lapisan teks digital. '
                'Aktifkan OCR_BACKEND=vllm atau OCR_BACKEND=transformers.'
            )
        return extracted

    model_id = os.getenv('OCR_MODEL_ID', DEFAULT_OCR_MODEL).strip() or DEFAULT_OCR_MODEL
    prompt = os.getenv('OCR_PROMPT', DEFAULT_OCR_PROMPT).strip() or DEFAULT_OCR_PROMPT

    if mode == 'transformers':
        parsed = _parse_via_transformers(
            pages, model_id=model_id, prompt=prompt, work_dir=work_dir
        )
    else:
        base_url = os.getenv('OCR_BASE_URL', DEFAULT_OCR_BASE_URL).strip() or DEFAULT_OCR_BASE_URL
        api_key = os.getenv('OCR_API_KEY', '').strip()
        max_tokens = int(os.getenv('OCR_MAX_TOKENS', '4096'))
        timeout = float(os.getenv('OCR_TIMEOUT_SECONDS', '180'))
        parsed = []
        for page in pages:
            md = _parse_page_via_vllm(
                page.image_path,
                base_url=base_url,
                model_id=model_id,
                api_key=api_key,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            if md:
                parsed.append(PageMarkdown(page_number=page.page_number, markdown=md))

    if not parsed:
        raise DocumentValidationError('Tidak ada teks materi yang terbaca dari dokumen.')
    return parsed
