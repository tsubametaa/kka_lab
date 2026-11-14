"""FastAPI endpoints for memory and knowledge management."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=['memory'])

class KnowledgeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=120000)
    answer: str = Field(min_length=1, max_length=32000)

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

