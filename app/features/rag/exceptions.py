"""RAG-related custom exceptions."""

class RagUnavailableError(RuntimeError):
    """Dilempar ketika layanan RAG Astra DB sedang tidak dapat diakses."""