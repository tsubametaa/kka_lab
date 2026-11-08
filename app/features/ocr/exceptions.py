"""Exceptions for document validation and Unlimited-OCR execution."""


class DocumentValidationError(ValueError):
    """Raised when an uploaded file is invalid, encrypted, empty, or exceeds limits."""


class OcrUnavailableError(RuntimeError):
    """Raised when the Unlimited-OCR backend cannot be reached or initialized."""

