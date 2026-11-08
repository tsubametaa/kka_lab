"""PDF and image page renderer using PyMuPDF for Unlimited-OCR input."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

from app.features.ocr.exceptions import DocumentValidationError, OcrUnavailableError

SUPPORTED_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
SUPPORTED_SUFFIXES = {'.pdf', *SUPPORTED_IMAGE_SUFFIXES}
DEFAULT_MAX_FILE_MB = 20
DEFAULT_MAX_PAGES = 30
DEFAULT_RENDER_DPI = 150


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_path: Path
    native_text: str = ''


def max_upload_bytes():
    mb = int(os.getenv('OCR_MAX_FILE_MB', str(DEFAULT_MAX_FILE_MB)))
    return max(1, mb) * 1024 * 1024


def max_document_pages():
    return max(1, int(os.getenv('OCR_MAX_PAGES', str(DEFAULT_MAX_PAGES))))


def validate_upload_metadata(filename, size_bytes, max_bytes=None):
    clean_name = Path(filename or '').name.strip()
    if not clean_name:
        raise DocumentValidationError('Nama file dokumen tidak valid.')
    suffix = Path(clean_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ', '.join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentValidationError(
            f'Format file {suffix or "(tanpa ekstensi)"} tidak didukung. Gunakan: {allowed}.'
        )
    if size_bytes <= 0:
        raise DocumentValidationError('File dokumen kosong.')
    limit = max_bytes if max_bytes is not None else max_upload_bytes()
    if size_bytes > limit:
        limit_mb = round(limit / (1024 * 1024), 1)
        raise DocumentValidationError(
            f'Ukuran file melebihi batas maksimum {limit_mb} MB.'
        )
    return clean_name, suffix


def render_document_pages(file_path, work_dir, *, dpi=None, max_pages=None):
    source = Path(file_path)
    if not source.exists() or source.stat().st_size <= 0:
        raise DocumentValidationError('File dokumen tidak ditemukan atau kosong.')

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentValidationError(f'Format file {suffix} tidak didukung.')

    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        target = out_dir / f'page_0001{suffix}'
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return [RenderedPage(page_number=1, image_path=target, native_text='')]

    if pymupdf is None:
        raise OcrUnavailableError(
            "Pustaka 'pymupdf' belum terpasang. Jalankan: pip install -r requirements-ocr.txt"
        )

    render_dpi = int(dpi or os.getenv('OCR_RENDER_DPI', str(DEFAULT_RENDER_DPI)))
    page_limit = max_pages if max_pages is not None else max_document_pages()

    try:
        doc = pymupdf.open(str(source))
    except Exception as exc:
        raise DocumentValidationError(f'File PDF rusak atau tidak dapat dibuka: {exc}') from exc

    try:
        if doc.is_encrypted:
            raise DocumentValidationError('File PDF terkunci password dan tidak dapat diproses.')
        total = len(doc)
        if total < 1:
            raise DocumentValidationError('File PDF tidak memiliki halaman.')
        if total > page_limit:
            raise DocumentValidationError(
                f'Jumlah halaman PDF ({total}) melebihi batas maksimum {page_limit} halaman.'
            )
        rendered = []
        for idx, page in enumerate(doc, start=1):
            native_text = (page.get_text('text') or '').strip()
            pix = page.get_pixmap(dpi=render_dpi)
            img_path = out_dir / f'page_{idx:04d}.png'
            pix.save(str(img_path))
            rendered.append(
                RenderedPage(page_number=idx, image_path=img_path, native_text=native_text)
            )
        return rendered
    finally:
        doc.close()

