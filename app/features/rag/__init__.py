"""RAG feature package: retrieval, chunking, scoped ingestion, auto-learning, and API routes."""

from app.features.rag.auto_learner import (
    AUTO_LEARN_SOURCE_TITLE,
    evaluate_kka_gate,
    is_auto_learn_enabled,
    match_kka_domain_terms,
    maybe_auto_learn_from_exchange,
    parse_and_validate_distilled_note,
    verify_and_distill_kka_note,
)
from app.features.rag.chunker import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_OVERLAP_TOKENS,
    DocumentChunk,
    chunk_markdown_pages,
)
from app.features.rag.context import (
    SHORT_QUESTION_WORDS,
    build_context,
    filter_relevant_history,
    is_follow_up_question,
    resolve_query_topic,
    retrieval_query,
)
from app.features.rag.exceptions import RagUnavailableError
from app.features.rag.ingestor import (
    VALID_SCOPES,
    build_astra_documents,
    delete_document_chunks,
    delete_user_rag_documents,
    ingest_chunks,
)
from app.features.rag.retriever import build_scope_filter, get_astra_collection, retrieve

__all__ = [
    'AUTO_LEARN_SOURCE_TITLE',
    'DEFAULT_CHUNK_TOKENS',
    'DEFAULT_OVERLAP_TOKENS',
    'DocumentChunk',
    'SHORT_QUESTION_WORDS',
    'RagUnavailableError',
    'VALID_SCOPES',
    'build_astra_documents',
    'build_context',
    'build_scope_filter',
    'chunk_markdown_pages',
    'delete_document_chunks',
    'delete_user_rag_documents',
    'evaluate_kka_gate',
    'filter_relevant_history',
    'get_astra_collection',
    'ingest_chunks',
    'is_auto_learn_enabled',
    'is_follow_up_question',
    'match_kka_domain_terms',
    'maybe_auto_learn_from_exchange',
    'parse_and_validate_distilled_note',
    'resolve_query_topic',
    'retrieval_query',
    'retrieve',
    'verify_and_distill_kka_note',
]
