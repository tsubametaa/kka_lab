"""Unlimited-OCR document rendering and Markdown extraction feature."""

from app.features.ocr.client import (
    DEFAULT_OCR_BASE_URL,
    DEFAULT_OCR_MODEL,
    PageMarkdown,
    clean_ocr_markdown,
    parse_pages_to_markdown,
)
from app.features.ocr.exceptions import DocumentValidationError, OcrUnavailableError
from app.features.ocr.pdf_renderer import (
    SUPPORTED_IMAGE_SUFFIXES,
    SUPPORTED_SUFFIXES,
    RenderedPage,
    max_document_pages,
    max_upload_bytes,
    render_document_pages,
    validate_upload_metadata,
)

__all__ = [
    'DEFAULT_OCR_BASE_URL',
    'DEFAULT_OCR_MODEL',
    'DocumentValidationError',
    'OcrUnavailableError',
    'PageMarkdown',
    'RenderedPage',
    'SUPPORTED_IMAGE_SUFFIXES',
    'SUPPORTED_SUFFIXES',
    'clean_ocr_markdown',
    'max_document_pages',
    'max_upload_bytes',
    'parse_pages_to_markdown',
    'render_document_pages',
    'validate_upload_metadata',
]

