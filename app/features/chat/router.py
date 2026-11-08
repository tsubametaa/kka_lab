"""FastAPI endpoints for interactive chat assistant."""

from contextlib import nullcontext
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from app.features.language.detector import detect_language
from app.features.language.translator import bilingual_queries
from app.features.rag.auto_learner import (
    evaluate_kka_gate,
    is_auto_learn_enabled,
    maybe_auto_learn_from_exchange,
)
from app.features.rag.context import build_context, retrieval_query

logger = logging.getLogger(__name__)
router = APIRouter(tags=['chat'])

class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=120000)
    session_id: str | None = None
    mode: str = 'umum'
    consent_to_training: bool = False

@router.post('/api/chat')
def chat(payload: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    if payload.mode not in ('umum', 'terstruktur'):
        raise HTTPException(status_code=422, detail='Mode harus umum atau terstruktur.')
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail='Pertanyaan tidak boleh kosong.')

    store = request.app.state.store
    assistant = request.app.state.assistant
    session_id = payload.session_id
    if session_id is None:
        session_id = store.create_session(payload.user_id)
    elif not store.session_belongs_to(session_id, payload.user_id):
        raise HTTPException(status_code=404, detail='Sesi tidak ditemukan.')

    repeated = store.find_repeated_question(
        payload.user_id, question, mode=payload.mode
    )
    history = store.history(session_id)
    lang = detect_language(question, history)
    memories = store.find_relevant_knowledge(payload.user_id, question)
    if repeated:
        memories.insert(0, {'question': question, 'answer': repeated['answer']})
    q_label, a_label = ('Question', 'Answer') if lang == 'en' else ('Pertanyaan', 'Jawaban')
    memory = '\n\n'.join(
        f'[M{index}] {q_label}: {item["question"]}\n'
        f'{a_label}: {item["answer"]}'
        for index, item in enumerate(memories, start=1)
    )
    try:
        from app.features.rag.retriever import retrieve
        previous_topic = history[-2]['content'] if len(history) >= 2 else ''
        query = retrieval_query(question, previous_topic)
        lock = getattr(assistant, 'gpu_lock', None) or nullcontext()
        with lock:
            search_queries = bilingual_queries(
                query, assistant.model, assistant.tokenizer
            )
        docs = retrieve(search_queries, user_id=payload.user_id)
        context, sources = build_context(docs, assistant.tokenizer)
        persona = store.get_persona(payload.user_id)
        answer = assistant.answer(
            question, payload.mode, history, context, memory, lang=lang,
            persona=persona,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception('Gagal memproses chat')
        raise HTTPException(status_code=503, detail='Layanan AI tidak tersedia.') from error
    memory_hit = repeated is not None

    training = store.save_exchange(
        payload.user_id, session_id, question, answer, payload.mode, sources,
        payload.consent_to_training, memory_replay=memory_hit,
    )
    if is_auto_learn_enabled():
        passed, _ = evaluate_kka_gate(
            question, answer, sources=sources, memory_hit=memory_hit
        )
        if passed:
            background_tasks.add_task(
                maybe_auto_learn_from_exchange,
                store,
                question=question,
                answer=answer,
                sources=sources,
                memory_hit=memory_hit,
                assistant=assistant,
            )
    return {
        'session_id': session_id,
        'answer': answer,
        'sources': sources,
        'memory_hit': memory_hit,
        'training': training,
    }
