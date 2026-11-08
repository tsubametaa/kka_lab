"""FastAPI endpoints for per-student answer persona."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.features.persona.validator import MAX_PERSONA_CHARS, PersonaError, clean_persona

router = APIRouter(tags=['persona'])


class PersonaRequest(BaseModel):
    # Batas mentah sedikit longgar; batas akhir dicek setelah normalisasi.
    persona: str = Field(default='', max_length=MAX_PERSONA_CHARS * 2)


@router.get('/api/users/{user_id}/persona')
def get_persona(user_id: str, request: Request):
    return {'persona': request.app.state.store.get_persona(user_id)}


@router.put('/api/users/{user_id}/persona')
def save_persona(user_id: str, payload: PersonaRequest, request: Request):
    try:
        persona = clean_persona(payload.persona)
    except PersonaError as error:
        raise HTTPException(
            status_code=422, detail={'reason': error.reason, 'message': str(error)}
        ) from error
    request.app.state.store.save_persona(user_id, persona)
    return {'saved': True, 'persona': persona}


@router.delete('/api/users/{user_id}/persona')
def delete_persona(user_id: str, request: Request):
    request.app.state.store.delete_persona(user_id)
    return {'deleted': True}

