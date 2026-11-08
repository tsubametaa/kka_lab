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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    app.state.store = ChatStore(os.getenv('CHAT_DB_PATH', 'data/chat_memory.sqlite3'))
    app.state.assistant = LocalAssistant()
    yield

app = FastAPI(title='Kyara AI', lifespan=lifespan)

origins = [
    origin.strip()
    for origin in os.getenv(
        'FRONTEND_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173'
    ).split(',')
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=['GET', 'POST', 'PUT', 'DELETE'],
    allow_headers=['Content-Type', 'X-API-Key', 'X-Admin-Key'],
)

@app.get('/api/health')
def health():
    return {'status': 'ok'}

app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(persona_router)
app.include_router(rag_router)

__all__ = [
    'app',
    'LocalAssistant',
    'ChatRequest',
    'KnowledgeRequest',
    'PersonaRequest',
    'health',
]
