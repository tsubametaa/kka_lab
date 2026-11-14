"""Astra DB (NoSQL) conversation and memory store entrypoint."""

from app.features.memory.astra_store import AstraChatStore


class ChatStore(AstraChatStore):
    """Store utama percakapan, memori personal, persona, dan metadata RAG di Astra DB."""

    backend = 'astradb'
