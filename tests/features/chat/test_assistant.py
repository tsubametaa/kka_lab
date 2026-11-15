"""Tests for app.features.chat.assistant."""

from types import SimpleNamespace
import unittest

from app.features.chat.assistant import _resolve_local_load_options


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


if __name__ == '__main__':
    unittest.main()
