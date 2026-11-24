"""Persona feature package: student-defined answer style (tone only)."""

from app.features.persona.extractor import (
    PersonaCommand,
    aresolve_persona_command,
    build_persona_confirmation_context,
    build_persona_extractor_messages,
    build_persona_rejection_reply,
    detect_persona_command,
    is_auto_persona_enabled,
    is_auto_persona_llm_enabled,
    might_be_persona_instruction,
    parse_llm_persona_command,
    resolve_persona_command,
)
from app.features.persona.validator import MAX_PERSONA_CHARS, PersonaError, clean_persona


def __getattr__(name):
    if name in ('persona_router', 'PersonaRequest'):
        from app.features.persona import router as _router
        return _router.router if name == 'persona_router' else _router.PersonaRequest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'MAX_PERSONA_CHARS',
    'PersonaCommand',
    'PersonaError',
    'aresolve_persona_command',
    'build_persona_confirmation_context',
    'build_persona_extractor_messages',
    'build_persona_rejection_reply',
    'clean_persona',
    'detect_persona_command',
    'is_auto_persona_enabled',
    'is_auto_persona_llm_enabled',
    'might_be_persona_instruction',
    'parse_llm_persona_command',
    'persona_router',
    'PersonaRequest',
    'resolve_persona_command',
]

