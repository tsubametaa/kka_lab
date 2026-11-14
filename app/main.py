"""FastAPI application entrypoint for the Qwen + Astra DB RAG assistant."""

import logging
import os
import warnings
from contextlib import asynccontextmanager

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False
from fastapi import Depends, FastAPI

from app.core.security import verify_internal_api_key
from app.features.chat.assistant import LocalAssistant
from app.features.chat.router import ChatRequest, router as chat_router
from app.features.memory.router import KnowledgeRequest, router as memory_router
from app.features.memory.store import ChatStore
from app.features.persona.router import PersonaRequest, router as persona_router
from app.features.rag.router import router as rag_router

warnings.filterwarnings('ignore', message=r'.*_check_is_size.*', category=FutureWarning)

load_dotenv()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = ChatStore()
    app.state.assistant = LocalAssistant()
    yield

app = FastAPI(title='Kyara AI', lifespan=lifespan)

@app.get('/api/health')
def health():
    return {'status': 'ok'}

internal_auth = [Depends(verify_internal_api_key)]
app.include_router(chat_router, dependencies=internal_auth)
app.include_router(memory_router, dependencies=internal_auth)
app.include_router(persona_router, dependencies=internal_auth)
app.include_router(rag_router, dependencies=internal_auth)

__all__ = [
    'app',
    'LocalAssistant',
    'ChatRequest',
    'KnowledgeRequest',
    'PersonaRequest',
    'health',
]
