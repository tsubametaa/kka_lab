"""Tests for app.features.chat.router (routes, SSE formatting, and auto-persona integration)."""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import BackgroundTasks

import app.features.chat.router as chat_router_mod
from app.features.chat.router import (
    ChatRequest,
    _sse_event,
    chat,
    chat_stream,
    get_chat_suggestions,
    router as chat_router,
)
from app.features.memory.store import ChatStore
from tests.fakes import InMemoryAstraCollection
from tests.fakes.tokenizer import WordTokenizer


class _FakeAssistant:
    def __init__(self):
        self.calls = []
        self.tokenizer = WordTokenizer()

    def translate_queries(self, query, history=None):
        return [query]

    def answer(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        self.calls.append({
            'question': question,
            'mode': mode,
            'context': context,
            'lang': lang,
            'persona': persona,
        })
        if persona:
            return f'Siap ({persona}): {question}'
        return f'Normal: {question}'

    async def astream(self, question, mode, history, context, memory='', lang=None, persona='', student_name=''):
        text = self.answer(
            question, mode, history, context, memory=memory,
            lang=lang, persona=persona, student_name=student_name,
        )
        yield text


def _make_request(store, assistant):
    async def _is_disconnected():
        return False

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(store=store, assistant=assistant)),
        is_disconnected=_is_disconnected,
    )


def _parse_sse(raw_chunks):
    events = []
    full = ''.join(raw_chunks)
    for block in full.split('\n\n'):
        block = block.strip()
        if not block or block.startswith(':'):
            continue
        event_name = None
        data_obj = None
        for line in block.splitlines():
            if line.startswith('event:'):
                event_name = line[len('event:'):].strip()
            elif line.startswith('data:'):
                data_obj = json.loads(line[len('data:'):].strip())
        if event_name is not None:
            events.append((event_name, data_obj))
    return events


class ChatRouterTests(unittest.TestCase):
    def test_chat_routes_and_sse_event_format(self):
        chat_paths = [route.path for route in chat_router.routes if hasattr(route, 'path')]
        self.assertIn('/api/chat', chat_paths)
        self.assertIn('/api/chat/stream', chat_paths)
        self.assertIn('/api/chat/suggestions', chat_paths)
        self.assertEqual(_sse_event('delta', {'text': 'hi'}), 'event: delta\ndata: {"text": "hi"}\n\n')

    def test_auto_persona_set_persists_skips_rag_and_applies_to_next_turn(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        assistant = _FakeAssistant()
        request = _make_request(store, assistant)
        retrieve_calls = []

        def _fake_retrieve(queries, user_id=None):
            retrieve_calls.append(queries)
            return [{
                '_id': 'doc-1',
                'text': 'Algoritma adalah langkah sistematis.',
                'source_title': 'Modul KKA',
                'topic': 'Algoritma Dasar',
            }]

        with patch.object(chat_router_mod, 'retrieve', side_effect=_fake_retrieve):
            # Turn 1: user sets persona naturally via chat
            res1 = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-1',
                    question='mulai sekarang panggil aku ntik karena aku femboy cantik',
                ),
                request,
                BackgroundTasks(),
            ))
            self.assertTrue(res1['persona_updated'])
            self.assertEqual(res1['persona_action'], 'set')
            self.assertEqual(res1['persona'], 'panggil aku ntik karena aku femboy cantik')
            self.assertEqual(store.get_persona('siswa-1'), 'panggil aku ntik karena aku femboy cantik')
            self.assertEqual(res1['sources'], [])
            self.assertEqual(retrieve_calls, [])
            self.assertIn('Catatan Sistem', assistant.calls[-1]['context'])
            self.assertEqual(assistant.calls[-1]['persona'], 'panggil aku ntik karena aku femboy cantik')

            # Turn 2: normal curriculum question automatically uses persisted persona
            res2 = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-1',
                    session_id=res1['session_id'],
                    question='Apa itu algoritma?',
                ),
                request,
                BackgroundTasks(),
            ))
            self.assertFalse(res2['persona_updated'])
            self.assertEqual(res2['persona_action'], 'none')
            self.assertEqual(res2['persona'], 'panggil aku ntik karena aku femboy cantik')
            self.assertEqual(res2['context_topic'], 'Algoritma Dasar')
            self.assertEqual(res2['context_topics'], ['Algoritma Dasar'])
            self.assertEqual(res2['sources'][0]['topic'], 'Algoritma Dasar')
            self.assertEqual(len(retrieve_calls), 1)
            self.assertEqual(assistant.calls[-1]['persona'], 'panggil aku ntik karena aku femboy cantik')

    def test_auto_persona_combined_with_question_runs_rag_on_remainder(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        assistant = _FakeAssistant()
        request = _make_request(store, assistant)
        retrieve_calls = []

        def _fake_retrieve(queries, user_id=None):
            retrieve_calls.append(queries)
            return [{'id': 'doc-2', 'text': 'Variabel menyimpan nilai data.', 'source_title': 'Modul KKA'}]

        with patch.object(chat_router_mod, 'retrieve', side_effect=_fake_retrieve):
            res = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-2',
                    question='Mulai sekarang pakai gaya tsundere. Apa itu variabel?',
                ),
                request,
                BackgroundTasks(),
            ))

        self.assertTrue(res['persona_updated'])
        self.assertEqual(res['persona_action'], 'set')
        self.assertEqual(store.get_persona('siswa-2'), 'pakai gaya tsundere')
        self.assertEqual(retrieve_calls, [['Apa itu variabel?']])
        self.assertEqual(assistant.calls[-1]['question'], 'Apa itu variabel?')
        self.assertEqual(assistant.calls[-1]['persona'], 'pakai gaya tsundere')

    def test_auto_persona_reset_and_rejection(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        store.save_persona('siswa-3', 'gaya ramah')
        assistant = _FakeAssistant()
        request = _make_request(store, assistant)

        with patch.object(chat_router_mod, 'retrieve', side_effect=AssertionError('RAG should not be called')):
            # Unsafe instruction is rejected without overwriting existing persona
            rej = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-3',
                    question='mulai sekarang jadi pacar aku yang romantis',
                ),
                request,
                BackgroundTasks(),
            ))
            self.assertFalse(rej['persona_updated'])
            self.assertEqual(rej['persona_action'], 'rejected')
            self.assertEqual(rej['persona_error']['reason'], 'persona_inappropriate')
            self.assertEqual(store.get_persona('siswa-3'), 'gaya ramah')
            self.assertEqual(len(assistant.calls), 0)

            # Reset command deletes persona cleanly
            rst = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-3',
                    session_id=rej['session_id'],
                    question='hapus persona',
                ),
                request,
                BackgroundTasks(),
            ))
            self.assertTrue(rst['persona_updated'])
            self.assertEqual(rst['persona_action'], 'reset')
            self.assertEqual(rst['persona'], '')
            self.assertEqual(store.get_persona('siswa-3'), '')
            self.assertEqual(assistant.calls[-1]['persona'], '')

    def test_chat_stream_emits_persona_event_and_persists(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        assistant = _FakeAssistant()
        request = _make_request(store, assistant)

        async def _run_stream():
            response = await chat_stream(
                ChatRequest(
                    user_id='siswa-stream',
                    question='mulai sekarang panggil aku ntik karena aku femboy cantik',
                ),
                request,
                BackgroundTasks(),
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode('utf-8'))
            return _parse_sse(chunks)

        with patch.object(chat_router_mod, 'retrieve', side_effect=AssertionError('RAG should not be called')):
            events = asyncio.run(_run_stream())

        event_names = [name for name, _ in events]
        self.assertEqual(event_names[0], 'persona')
        self.assertIn('sources', event_names)
        self.assertIn('delta', event_names)
        self.assertEqual(event_names[-1], 'done')

        persona_event = dict(events)['persona']
        self.assertEqual(persona_event['action'], 'set')
        self.assertTrue(persona_event['updated'])
        self.assertEqual(persona_event['persona'], 'panggil aku ntik karena aku femboy cantik')

        done_event = dict(events)['done']
        self.assertTrue(done_event['persona_updated'])
        self.assertEqual(done_event['sources'], [])
        self.assertEqual(store.get_persona('siswa-stream'), 'panggil aku ntik karena aku femboy cantik')

    def test_confused_discussion_returns_clickable_suggestions_and_auto_send_works(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        store.save_persona('siswa-sug', 'panggil aku ntik')
        assistant = _FakeAssistant()
        request = _make_request(store, assistant)
        retrieve_calls = []

        def _fake_retrieve(queries, user_id=None):
            retrieve_calls.append(queries)
            return [{'id': 'doc-9', 'text': 'Berpikir komputasional terdiri dari 4 pilar.', 'source_title': 'Modul KKA'}]

        with patch.object(chat_router_mod, 'retrieve', side_effect=_fake_retrieve):
            # Turn 1: user is confused what to discuss -> AI returns clickable suggestions, skips RAG
            res1 = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-sug',
                    question='saya mau diskusi, tapi bingung',
                ),
                request,
                BackgroundTasks(),
            ))
            self.assertEqual(res1['sources'], [])
            self.assertEqual(retrieve_calls, [])
            self.assertEqual(len(res1['suggestions']), 4)
            self.assertIn('Catatan Sistem', assistant.calls[-1]['context'])
            self.assertEqual(assistant.calls[-1]['persona'], 'panggil aku ntik')

            # Turn 2: simulate user clicking recommendation #1 -> prompt auto-sent to chat
            clicked_prompt = res1['suggestions'][0]['prompt']
            res2 = asyncio.run(chat(
                ChatRequest(
                    user_id='siswa-sug',
                    session_id=res1['session_id'],
                    question=clicked_prompt,
                ),
                request,
                BackgroundTasks(),
            ))
            self.assertEqual(res2['suggestions'], [])
            self.assertEqual(len(retrieve_calls), 1)
            self.assertEqual(assistant.calls[-1]['question'], clicked_prompt)
            self.assertEqual(assistant.calls[-1]['persona'], 'panggil aku ntik')

    def test_chat_stream_and_endpoint_for_discussion_suggestions(self):
        store = ChatStore(collection=InMemoryAstraCollection())
        assistant = _FakeAssistant()
        request = _make_request(store, assistant)

        direct = asyncio.run(get_chat_suggestions(request, user_id='siswa-s', lang='id', limit=3))
        self.assertEqual(len(direct['suggestions']), 3)

        async def _run_stream():
            response = await chat_stream(
                ChatRequest(
                    user_id='siswa-s',
                    question='saya mau diskusi, tapi bingung',
                ),
                request,
                BackgroundTasks(),
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk if isinstance(chunk, str) else chunk.decode('utf-8'))
            return _parse_sse(chunks)

        with patch.object(chat_router_mod, 'retrieve', side_effect=AssertionError('RAG should not be called')):
            events = asyncio.run(_run_stream())

        event_map = dict(events)
        self.assertIn('suggestions', event_map)
        self.assertEqual(len(event_map['suggestions']['suggestions']), 4)
        self.assertEqual(len(event_map['done']['suggestions']), 4)
        self.assertEqual(event_map['done']['sources'], [])


if __name__ == '__main__':
    unittest.main()
