"""Chat feature package: prompt crafting, assistant inference, and API router."""

from app.features.chat.prompts import (
    BASE,
    DEFAULT_MAX_PROMPT_TOKENS,
    FORMAT,
    MODEL_MAX_CONTEXT_TOKENS,
    PEDAGOGY,
    POLICIES,
    count_prompt_tokens,
    fit_messages_to_budget,
    make_system,
    resolve_max_prompt_tokens,
)

def __getattr__(name):
    if name == 'LocalAssistant':
        from app.features.chat.assistant import LocalAssistant
        return LocalAssistant
    if name == 'ChatRequest':
        from app.features.chat.router import ChatRequest
        return ChatRequest
    if name == 'chat_router':
        from app.features.chat.router import router as chat_router
        return chat_router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BASE',
    'PEDAGOGY',
    'FORMAT',
    'POLICIES',
    'MODEL_MAX_CONTEXT_TOKENS',
    'DEFAULT_MAX_PROMPT_TOKENS',
    'resolve_max_prompt_tokens',
    'make_system',
    'count_prompt_tokens',
    'fit_messages_to_budget',
    'LocalAssistant',
    'ChatRequest',
    'chat_router',
]
