"""FastAPI endpoints for interactive chat assistant (JSON & SSE streaming)."""

import asyncio
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.features.chat.sanitizer import StreamingSanitizer, sanitize_response
from app.features.language.detector import detect_language
from app.features.language.translator import bilingual_queries
from app.features.rag.auto_learner import (
    evaluate_kka_gate,
    is_auto_learn_enabled,
    maybe_auto_learn_from_exchange,
)
from app.features.rag.context import (
    build_context,
    filter_relevant_history,
    is_document_reference_query,
    retrieval_query,
)
from app.features.rag.retriever import retrieve, retrieve_for_chat

logger = logging.getLogger(__name__)
router = APIRouter(tags=['chat'])


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    student_name: str | None = Field(default=None, max_length=120)
    question: str = Field(min_length=1, max_length=120000)
    session_id: str | None = None
    mode: str = 'umum'
    consent_to_training: bool = False
    reset_context: bool = False


def _sse_event(event_name: str, data: dict) -> str:
    return f'event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'


def _prepare_session_and_memory(store, payload: ChatRequest, question: str):
    session_id = payload.session_id
    if session_id is None:
        session_id = store.create_session(payload.user_id)
    elif not store.session_belongs_to(session_id, payload.user_id):
        raise HTTPException(status_code=404, detail='Sesi tidak ditemukan.')

    repeated = (
        None
        if is_document_reference_query(question)
        else store.find_repeated_question(payload.user_id, question, mode=payload.mode)
    )
    history = store.history(session_id)
    active_history = [] if payload.reset_context else filter_relevant_history(question, history)
    lang = detect_language(question, active_history)
    memories = store.find_relevant_knowledge(payload.user_id, question)
    if repeated:
        memories.insert(0, {'question': question, 'answer': repeated['answer']})
    q_label, a_label = ('Question', 'Answer') if lang == 'en' else ('Pertanyaan', 'Jawaban')
    mem_prefix = 'Note' if lang == 'en' else 'Riwayat'
    memory = '\n\n'.join(
        f'{mem_prefix} #{index} — {q_label}: {item["question"]}\n{a_label}: {item["answer"]}'
        for index, item in enumerate(memories, start=1)
    )
    return session_id, active_history, lang, memory, repeated is not None


def _is_recent_job(job, max_age_seconds=180):
    raw_ts = job.get('updated_at') or job.get('created_at') or ''
    if not raw_ts:
        return False
    try:
        dt = datetime.fromisoformat(str(raw_ts).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() <= max_age_seconds
    except Exception:
        return False


def _pick_target_user_doc_id(question, completed_docs):
    if not completed_docs:
        return None
    lower = (question or '').lower()
    for doc in completed_docs:
        for field in ('source_title', 'filename'):
            val = str(doc.get(field) or '').strip().lower()
            stem = val.rsplit('.', 1)[0] if '.' in val else val
            if len(stem) >= 4 and stem in lower:
                return doc.get('doc_id')
    return completed_docs[0].get('doc_id')


async def _wait_for_active_user_jobs(store, active_jobs, max_wait_seconds=12.0):
    if not active_jobs or not hasattr(store, 'get_ingestion_job'):
        return
    recent_active = [j for j in active_jobs if j.get('job_id') and _is_recent_job(j)]
    if not recent_active:
        return
    deadline = asyncio.get_running_loop().time() + max_wait_seconds
    target_id = recent_active[0]['job_id']
    while asyncio.get_running_loop().time() < deadline:
        job = await run_in_threadpool(store.get_ingestion_job, target_id)
        if not job or job.get('status') not in ('queued', 'processing'):
            return
        await asyncio.sleep(1.5)


async def _prepare_rag_and_persona(assistant, store, payload: ChatRequest, question: str, active_history):
    previous_user = ''
    previous_assistant = ''
    for msg in reversed(active_history or []):
        role = msg.get('role')
        if role == 'assistant' and not previous_assistant:
            previous_assistant = msg.get('content', '')
        elif role == 'user' and not previous_user:
            previous_user = msg.get('content', '')
        if previous_user and previous_assistant:
            break

    user_docs = []
    if hasattr(store, 'list_user_documents'):
        user_docs = await run_in_threadpool(store.list_user_documents, payload.user_id)

    explicit_doc_ref = is_document_reference_query(question, user_docs)
    completed_docs = [
        d for d in user_docs
        if d.get('status') == 'completed' and (d.get('chunks_inserted') or 0) > 0
    ]
    active_jobs = [d for d in user_docs if d.get('status') in ('queued', 'processing')]
    if active_jobs and (explicit_doc_ref or not completed_docs):
        await _wait_for_active_user_jobs(store, active_jobs)
        user_docs = await run_in_threadpool(store.list_user_documents, payload.user_id)
        completed_docs = [
            d for d in user_docs
            if d.get('status') == 'completed' and (d.get('chunks_inserted') or 0) > 0
        ]
        active_jobs = [d for d in user_docs if d.get('status') in ('queued', 'processing')]
        explicit_doc_ref = is_document_reference_query(question, user_docs)

    query = retrieval_query(question, previous_user, previous_assistant)
    recent_turn = active_history[-2:] if active_history else None
    if hasattr(assistant, 'atranslate_queries'):
        search_queries = await assistant.atranslate_queries(query, history=recent_turn)
    elif hasattr(assistant, 'translate_queries'):
        search_queries = await run_in_threadpool(assistant.translate_queries, query, recent_turn)
    else:
        def _sync_translate():
            lock = getattr(assistant, 'gpu_lock', None) or nullcontext()
            with lock:
                return bilingual_queries(query, assistant.model, assistant.tokenizer, history=recent_turn)
        search_queries = await run_in_threadpool(_sync_translate)

    target_doc_id = _pick_target_user_doc_id(question, completed_docs)
    if completed_docs:
        docs = await run_in_threadpool(
            retrieve_for_chat,
            search_queries,
            user_id=payload.user_id,
            latest_doc_id=target_doc_id,
            explicit_doc_ref=explicit_doc_ref,
            top_k=5,
            user_top_k=5,
        )
        context_budget = 2200
    else:
        docs = await run_in_threadpool(retrieve, search_queries, user_id=payload.user_id)
        context_budget = 1200

    context, sources = await run_in_threadpool(
        build_context, docs, assistant.tokenizer, context_budget
    )

    if explicit_doc_ref and not any(
        s.get('scope') == 'user' for s in sources
    ):
        recent_active = [j for j in active_jobs if _is_recent_job(j)]
        if recent_active:
            fname = recent_active[0].get('filename') or 'dokumen'
            note = (
                f'Catatan Status Unggahan: Dokumen "{fname}" masih sedang diproses OCR. '
                'Beri tahu siswa untuk menunggu beberapa detik lalu tanyakan kembali.'
            )
            context = f'{note}\n\n{context}'.strip() if context else note
        elif user_docs and user_docs[0].get('status') == 'failed' and _is_recent_job(user_docs[0], max_age_seconds=900):
            fname = user_docs[0].get('filename') or 'dokumen'
            err = user_docs[0].get('error_message') or 'gagal diproses'
            note = (
                f'Catatan Status Unggahan: Dokumen "{fname}" gagal diproses ({err}). '
                'Sampaikan jujur bahwa dokumen belum terbaca dan sarankan unggah ulang.'
            )
            context = f'{note}\n\n{context}'.strip() if context else note

    persona = await run_in_threadpool(store.get_persona, payload.user_id)
    return context, sources, persona


def _finalize_exchange(store, assistant, background_tasks: BackgroundTasks, payload: ChatRequest,
                       session_id: str, question: str, answer: str, sources, memory_hit: bool):
    training = store.save_exchange(
        payload.user_id, session_id, question, answer, payload.mode, sources,
        payload.consent_to_training, memory_replay=memory_hit,
    )
    if is_auto_learn_enabled():
        passed, _ = evaluate_kka_gate(question, answer, sources=sources, memory_hit=memory_hit)
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
    return training


@router.post('/api/chat')
async def chat(payload: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    if payload.mode not in ('umum', 'terstruktur'):
        raise HTTPException(status_code=422, detail='Mode harus umum atau terstruktur.')
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail='Pertanyaan tidak boleh kosong.')

    store = request.app.state.store
    assistant = request.app.state.assistant
    session_id, active_history, lang, memory, memory_hit = await run_in_threadpool(
        _prepare_session_and_memory, store, payload, question
    )
    try:
        context, sources, persona = await _prepare_rag_and_persona(
            assistant, store, payload, question, active_history
        )
        student_name = payload.student_name or ''
        if hasattr(assistant, 'aanswer'):
            answer = await assistant.aanswer(
                question, payload.mode, active_history, context, memory,
                lang=lang, persona=persona, student_name=student_name,
            )
        else:
            answer = await run_in_threadpool(
                assistant.answer,
                question, payload.mode, active_history, context, memory,
                lang=lang, persona=persona, student_name=student_name,
            )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception('Gagal memproses chat')
        raise HTTPException(status_code=503, detail='Layanan AI tidak tersedia.') from error

    answer = sanitize_response(answer)
    training = await run_in_threadpool(
        _finalize_exchange, store, assistant, background_tasks,
        payload, session_id, question, answer, sources, memory_hit,
    )
    return {
        'session_id': session_id,
        'answer': answer,
        'sources': sources,
        'memory_hit': memory_hit,
        'training': training,
    }

@router.post('/api/chat/stream')
async def chat_stream(payload: ChatRequest, request: Request, background_tasks: BackgroundTasks):
    if payload.mode not in ('umum', 'terstruktur'):
        raise HTTPException(status_code=422, detail='Mode harus umum atau terstruktur.')
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail='Pertanyaan tidak boleh kosong.')

    store = request.app.state.store
    assistant = request.app.state.assistant
    session_id, active_history, lang, memory, memory_hit = await run_in_threadpool(
        _prepare_session_and_memory, store, payload, question
    )
    student_name = payload.student_name or ''

    async def _event_generator():
        yield ': retrieving\n\n'
        chunks = []
        stream_sanitizer = StreamingSanitizer()
        try:
            context, sources, persona = await _prepare_rag_and_persona(
                assistant, store, payload, question, active_history
            )
            yield _sse_event('sources', {'session_id': session_id, 'sources': sources})

            if hasattr(assistant, 'astream'):
                async for delta in assistant.astream(
                    question, payload.mode, active_history, context, memory,
                    lang=lang, persona=persona, student_name=student_name,
                ):
                    if await request.is_disconnected():
                        return
                    cleaned = stream_sanitizer.feed(delta)
                    if cleaned:
                        chunks.append(cleaned)
                        yield _sse_event('delta', {'text': cleaned})
                tail = stream_sanitizer.flush()
                if tail:
                    chunks.append(tail)
                    yield _sse_event('delta', {'text': tail})
            else:
                answer = await run_in_threadpool(
                    assistant.answer,
                    question, payload.mode, active_history, context, memory,
                    lang=lang, persona=persona, student_name=student_name,
                )
                cleaned_answer = sanitize_response(answer)
                chunks.append(cleaned_answer)
                yield _sse_event('delta', {'text': cleaned_answer})

            full_answer = sanitize_response(''.join(chunks))
            training = await run_in_threadpool(
                _finalize_exchange, store, assistant, background_tasks,
                payload, session_id, question, full_answer, sources, memory_hit,
            )
            yield _sse_event('done', {
                'session_id': session_id,
                'answer': full_answer,
                'sources': sources,
                'memory_hit': memory_hit,
                'training': training,
            })
        except asyncio.CancelledError:
            return
        except ValueError as error:
            yield _sse_event('error', {'detail': str(error)})
        except Exception:
            logger.exception('Gagal memproses stream chat')
            yield _sse_event('error', {'detail': 'Layanan AI tidak tersedia.'})

    return StreamingResponse(
        _event_generator(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )
