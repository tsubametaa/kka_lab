"""Tests to verify layered by feature architecture imports and integrity."""

import unittest

from app.core import load_record
from app.core.config import load_record as core_load_record
from app.features.chat import (
    BASE,
    DEFAULT_MAX_PROMPT_TOKENS,
    FORMAT,
    MODEL_MAX_CONTEXT_TOKENS,
    PEDAGOGY,
    POLICIES,
    ChatRequest,
    fit_messages_to_budget,
    make_system,
    resolve_max_prompt_tokens,
)
from app.features.chat.prompts import (
    DEFAULT_MAX_PROMPT_TOKENS as chat_default_max_prompt,
    MODEL_MAX_CONTEXT_TOKENS as chat_model_max_ctx,
    fit_messages_to_budget as chat_fit_messages,
    make_system as chat_make_system,
    resolve_max_prompt_tokens as chat_resolve_max_prompt,
)
from app.features.dataset import encode_target, split_groups, validate_rows
from app.features.dataset.tokenizer import split_groups as dataset_split_groups
from app.features.dataset.validator import validate_rows as dataset_validate_rows
from app.features.language import (
    LOCALES,
    bilingual_queries,
    detect_language,
    ensure_response_language,
    format_user_message,
    is_wrong_language,
    parse_bilingual_queries,
)
from app.features.language.detector import detect_language as detector_detect_lang
from app.features.memory import (
    ChatStore,
    KnowledgeRequest,
    normalize_question,
    training_quality,
)
from app.features.memory.gate import training_quality as gate_training_quality
from app.features.memory.store import ChatStore as memory_chat_store
from app.features.rag import (
    DocumentChunk,
    RagUnavailableError,
    build_astra_documents,
    build_context,
    build_scope_filter,
    chunk_markdown_pages,
    ingest_chunks,
    resolve_query_topic,
    retrieval_query,
    retrieve,
)
from app.features.ocr import (
    DocumentValidationError,
    OcrUnavailableError,
    PageMarkdown,
    RenderedPage,
    clean_ocr_markdown,
    parse_pages_to_markdown,
    render_document_pages,
)
from app.features.rag.context import retrieval_query as rag_retrieval_query
from app.main import app, health


class ArchitectureTests(unittest.TestCase):
    def test_core_exports(self):
        self.assertIs(load_record, core_load_record)

    def test_chat_feature_exports(self):
        self.assertIs(make_system, chat_make_system)
        self.assertIs(fit_messages_to_budget, chat_fit_messages)
        self.assertIs(resolve_max_prompt_tokens, chat_resolve_max_prompt)
        self.assertEqual(DEFAULT_MAX_PROMPT_TOKENS, chat_default_max_prompt)
        self.assertEqual(MODEL_MAX_CONTEXT_TOKENS, chat_model_max_ctx)
        self.assertEqual(MODEL_MAX_CONTEXT_TOKENS, 32768)
        self.assertEqual(DEFAULT_MAX_PROMPT_TOKENS, 32224)
        prompt = make_system('umum', 'test context')
        self.assertIn('test context', prompt)
        self.assertTrue(issubclass(ChatRequest, object))
        long_req = ChatRequest(user_id='siswa1', question='a' * 50000)
        self.assertEqual(len(long_req.question), 50000)

    def test_dataset_feature_exports(self):
        self.assertIs(split_groups, dataset_split_groups)
        self.assertIs(validate_rows, dataset_validate_rows)

    def test_language_feature_exports(self):
        self.assertIs(detect_language, detector_detect_lang)
        self.assertEqual(detect_language('What is machine learning?'), 'en')
        self.assertEqual(detect_language('Apa itu machine learning?'), 'id')

    def test_memory_feature_exports(self):
        self.assertIs(ChatStore, memory_chat_store)
        self.assertIs(training_quality, gate_training_quality)
        self.assertTrue(issubclass(KnowledgeRequest, object))

    def test_rag_feature_exports(self):
        self.assertIs(retrieval_query, rag_retrieval_query)
        self.assertEqual(retrieval_query('contohnya?', 'apa itu ai?'), 'apa itu ai? contohnya?')
        self.assertTrue(callable(chunk_markdown_pages))
        self.assertTrue(callable(build_astra_documents))
        self.assertTrue(callable(ingest_chunks))
        self.assertIn('$or', build_scope_filter('siswa-1'))
        self.assertTrue(issubclass(DocumentChunk, object))

    def test_ocr_feature_exports(self):
        self.assertTrue(issubclass(DocumentValidationError, ValueError))
        self.assertTrue(issubclass(OcrUnavailableError, RuntimeError))
        self.assertTrue(issubclass(PageMarkdown, object))
        self.assertTrue(issubclass(RenderedPage, object))
        self.assertTrue(callable(parse_pages_to_markdown))
        self.assertTrue(callable(render_document_pages))
        self.assertEqual(clean_ocr_markdown('Halo<｜end▁of▁sentence｜>'), 'Halo')

    def test_web_api_routes_inclusion(self):
        from app.features.chat.router import router as chat_router
        from app.features.memory.router import router as memory_router
        from app.features.rag.router import router as rag_router

        chat_routes = [route.path for route in chat_router.routes if hasattr(route, 'path')]
        memory_routes = [route.path for route in memory_router.routes if hasattr(route, 'path')]
        rag_routes = [route.path for route in rag_router.routes if hasattr(route, 'path')]
        api_routes = [route.path for route in app.routes if hasattr(route, 'path')]

        self.assertIn('/api/health', api_routes)
        self.assertIn('/api/chat', chat_routes)
        self.assertIn('/api/users/{user_id}/knowledge', memory_routes)
        self.assertIn('/api/users/{user_id}/memory', memory_routes)
        self.assertIn('/api/rag/documents', rag_routes)
        self.assertIn('/api/rag/jobs/{job_id}', rag_routes)
        self.assertIn('/api/users/{user_id}/documents', rag_routes)
        self.assertIn('/api/rag/auto-learned', rag_routes)
        self.assertIn('/api/rag/documents/{doc_id}', rag_routes)
        self.assertIn('/api/rag/documents', api_routes)
        from app.features.persona.router import router as persona_router
        persona_routes = [route.path for route in persona_router.routes if hasattr(route, 'path')]
        self.assertIn('/api/users/{user_id}/persona', persona_routes)
        self.assertEqual(health(), {'status': 'ok'})


if __name__ == '__main__':
    unittest.main()
