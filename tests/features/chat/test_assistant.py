"""Tests for app.features.chat.assistant."""

import asyncio
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import app.features.chat.assistant as assistant_mod
from app.features.chat.assistant import LocalAssistant, _resolve_local_load_options
from tests.fakes.tokenizer import ChatTemplateTokenizer


class _FakeTensor:
    def __init__(self, shape=(1, 5)):
        self.shape = shape

    def to(self, _device):
        return self


class _FakeStreamingTokenizer(ChatTemplateTokenizer):
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = ''.join(f'<{m["role"]}>{m["content"]}</end>' for m in messages)
        if add_generation_prompt:
            rendered += '<assistant>'
        return [ord(x) for x in rendered] if tokenize else rendered

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {'input_ids': _FakeTensor(), 'attention_mask': _FakeTensor()}


class _FakeStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        self._q = queue.Queue()
        self._stop = object()

    def put(self, text):
        self._q.put(text)

    def end(self):
        self._q.put(self._stop)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is self._stop:
                break
            yield item


class ChatAssistantTests(unittest.TestCase):
    def test_resolve_local_load_options_selects_bf16_or_4bit_by_vram(self):
        def _fake_torch(vram_gb, bf16):
            return SimpleNamespace(
                bfloat16='bf16',
                float16='fp16',
                cuda=SimpleNamespace(
                    is_bf16_supported=lambda: bf16,
                    get_device_properties=lambda _idx: SimpleNamespace(
                        total_memory=vram_gb * (1024 ** 3)
                    ),
                ),
            )

        self.assertEqual(_resolve_local_load_options(_fake_torch(48, True)), ('bf16', False))
        self.assertEqual(_resolve_local_load_options(_fake_torch(8, False)), ('fp16', True))

    def test_local_assistant_astream_streams_chunks_cleanly(self):
        class _FakeModel:
            def generate(self, **kwargs):
                self.received_kwargs = dict(kwargs)
                kwargs['streamer'].put('Langkah 1: ')
                kwargs['streamer'].put('Pahami masalah.')
                kwargs['streamer'].end()

        class _FakeInferenceMode:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

        fake_torch = SimpleNamespace(
            inference_mode=_FakeInferenceMode,
            cuda=SimpleNamespace(is_available=lambda: False),
        )

        inst = object.__new__(LocalAssistant)
        inst.tokenizer = _FakeStreamingTokenizer()
        inst.model = _FakeModel()
        inst.gpu_lock = threading.RLock()
        inst.max_history_turns = 4
        inst.max_new_tokens = 128
        inst.do_sample = False
        inst.temperature = 0.45
        inst.top_p = 0.9
        inst.min_p = 0.05
        inst.repetition_penalty = 1.08
        inst.max_prompt_tokens = 8192

        async def _collect():
            chunks = []
            async for delta in inst.astream('Apa itu AI?', 'umum', [], ''):
                chunks.append(delta)
            return chunks

        with patch.object(assistant_mod, 'TextIteratorStreamer', _FakeStreamer), \
             patch.object(assistant_mod, 'torch', fake_torch):
            result = asyncio.run(_collect())

        self.assertEqual(result, ['Langkah 1: ', 'Pahami masalah.'])
        self.assertIn('input_ids', inst.model.received_kwargs)
        self.assertIn('attention_mask', inst.model.received_kwargs)


if __name__ == '__main__':
    unittest.main()
