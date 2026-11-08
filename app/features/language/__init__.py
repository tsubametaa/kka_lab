"""Language feature package: detection, translation, and localized constants."""

from app.features.language import locales
from app.features.language.detector import (
    detect_language,
    format_user_message,
    is_wrong_language,
)
from app.features.language.locales import LOCALES, en, id
from app.features.language.translator import (
    bilingual_queries,
    ensure_response_language,
    parse_bilingual_queries,
)

__all__ = [
    'locales',
    'LOCALES',
    'id',
    'en',
    'detect_language',
    'format_user_message',
    'is_wrong_language',
    'bilingual_queries',
    'ensure_response_language',
    'parse_bilingual_queries',
]
