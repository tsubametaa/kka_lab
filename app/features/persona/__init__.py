"""Persona feature package: student-defined answer style (tone only)."""

from app.features.persona.validator import MAX_PERSONA_CHARS, PersonaError, clean_persona


def __getattr__(name):
    if name in ('persona_router', 'PersonaRequest'):
        from app.features.persona import router as _router
        return _router.router if name == 'persona_router' else _router.PersonaRequest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'MAX_PERSONA_CHARS',
    'PersonaError',
    'clean_persona',
    'persona_router',
    'PersonaRequest',
]

