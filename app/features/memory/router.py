"""FastAPI endpoints for memory, knowledge, and admin chat log management."""

import os
import secrets

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=['memory'])


class KnowledgeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=120000)
    answer: str = Field(min_length=1, max_length=32000)


def _require_admin_key(request: Request):
    admin_secret = os.getenv('RAG_ADMIN_SECRET', '').strip()
    if not admin_secret:
        return
    provided = request.headers.get('x-admin-key', '')
    if not secrets.compare_digest(provided, admin_secret):
        raise HTTPException(
            status_code=403,
            detail='Operasi admin memerlukan header X-Admin-Key yang valid.',
        )


@router.post('/api/users/{user_id}/knowledge')
def save_knowledge(user_id: str, payload: KnowledgeRequest, request: Request):
    question = payload.question.strip()
    answer = payload.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=422, detail='Pertanyaan dan jawaban tidak boleh kosong.')
    request.app.state.store.save_knowledge(user_id, question, answer)
    return {'saved': True}


@router.delete('/api/users/{user_id}/memory')
def delete_memory(user_id: str, request: Request):
    request.app.state.store.delete_user_data(user_id)
    return {'deleted': True}


@router.get('/api/admin/chats')
def list_admin_chat_logs(
    request: Request,
    user_id: str = Query(default=''),
    session_id: str = Query(default=''),
    limit: int = Query(default=50, ge=1, le=200),
):
    _require_admin_key(request)
    items = request.app.state.store.list_chat_exchanges(
        user_id=user_id.strip() or None,
        session_id=session_id.strip() or None,
        limit=limit,
    )
    return {'items': items}


@router.delete('/api/admin/chats/sessions/{session_id}')
def delete_admin_chat_session(session_id: str, request: Request):
    _require_admin_key(request)
    deleted = request.app.state.store.delete_session_data(session_id)
    return {'deleted': True, 'session_id': session_id, 'deleted_count': deleted}


@router.delete('/api/admin/chats/exchanges/{exchange_id}')
def delete_admin_chat_exchange(exchange_id: str, request: Request):
    _require_admin_key(request)
    deleted = request.app.state.store.delete_exchange_record(exchange_id)
    return {'deleted': True, 'exchange_id': exchange_id, 'deleted_count': deleted}
