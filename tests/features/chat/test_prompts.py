"""Tests for app.features.chat.prompts (system prompts, token budget, generation kwargs)."""

import unittest

from app.features.chat import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SAMPLING_CONFIG,
    MODEL_MAX_CONTEXT_TOKENS,
    build_generation_kwargs,
    count_prompt_tokens,
    fit_messages_to_budget,
    make_system,
    resolve_max_prompt_tokens,
)
from tests.fakes import ChatTemplateTokenizer


class ChatPromptsTests(unittest.TestCase):
    def test_make_system_modes_and_bilingual_rules(self):
        general = make_system('umum', 'x')
        structured = make_system('terstruktur', 'x')
        self.assertNotEqual(general, structured)
        self.assertIn('jangan memberikan jawaban akhir saja', general.lower())
        self.assertIn('secara proporsional', general)
        self.assertIn('[S1], [S2], [M1]', general)
        self.assertNotIn('90-160 kata', general)
        self.assertIn('DILARANG membuka jawaban dengan kalimat perintah kaku', general)
        self.assertIn('contoh konkret', general)
        self.assertIn('simulasi kasus sederhana', general)
        self.assertIn('SMA/SMK kelas X', general)
        self.assertIn('langkah demi langkah', general)
        self.assertIn('memantik rasa penasaran', general)
        self.assertIn('Jangan berikan solusi lengkap pada giliran awal', structured)
        self.assertIn('tanpa melompati satu langkah pun', structured)

        personal = make_system('umum', 'x', '[M1] catatan')
        self.assertIn('belum terverifikasi', personal)
        self.assertIn('[M1] catatan', personal)
        self.assertIn('English', personal)
        self.assertIn('bahasa yang sama', personal)
        self.assertIn('ATURAN BAHASA WAJIB', personal)

        general_en = make_system('umum', '[S1] Materi Indonesia', '[M1] note', lang='en')
        structured_en = make_system('terstruktur', '[S1] Materi Indonesia', lang='en')
        self.assertNotEqual(general_en, structured_en)
        self.assertIn('GENERAL MODE', general_en)
        self.assertIn('STRUCTURED MODE', structured_en)
        self.assertIn('proportionally', general_en)
        self.assertNotIn('90-160 words', general_en)
        self.assertIn('case simulation', general_en)
        self.assertIn('Grade 10', general_en)
        self.assertIn('SOURCE CONTEXT', general_en)
        self.assertIn('PERSONAL MEMORY', general_en)
        self.assertIn('MANDATORY LANGUAGE RULE', general_en)
        self.assertIn('Never answer in Indonesian.', general_en)
        with self.assertRaises(ValueError):
            make_system('umum', 'x', lang='fr')

    def test_build_generation_kwargs_sampling_and_greedy(self):
        class DummyTokenizer:
            pad_token_id = 0
            eos_token_id = 2

        tok = DummyTokenizer()
        sampled = build_generation_kwargs(tok)
        self.assertTrue(sampled['do_sample'])
        self.assertEqual(sampled['max_new_tokens'], DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(sampled['temperature'], DEFAULT_SAMPLING_CONFIG['temperature'])
        self.assertEqual(sampled['top_p'], DEFAULT_SAMPLING_CONFIG['top_p'])
        self.assertEqual(sampled['min_p'], DEFAULT_SAMPLING_CONFIG['min_p'])
        self.assertEqual(sampled['repetition_penalty'], DEFAULT_SAMPLING_CONFIG['repetition_penalty'])

        greedy = build_generation_kwargs(tok, do_sample=False, repetition_penalty=1.0)
        self.assertFalse(greedy['do_sample'])
        self.assertIsNone(greedy['temperature'])
        self.assertIsNone(greedy['top_p'])
        self.assertNotIn('min_p', greedy)

        with self.assertRaises(ValueError):
            build_generation_kwargs(tok, do_sample=True, temperature=0.0)

    def test_fit_messages_to_budget_normal(self):
        tok = ChatTemplateTokenizer()
        messages, tokens = fit_messages_to_budget(
            mode='terstruktur',
            context='[S1] Materi singkat',
            user_message='apa itu prompt?',
            history=[{'role': 'user', 'content': 'halo'}, {'role': 'assistant', 'content': 'halo juga'}],
            tokenizer=tok,
            max_prompt_tokens=4096,
        )
        self.assertEqual(len(messages), 4)  # system + 2 history + user
        self.assertLessEqual(tokens, 4096)

    def test_fit_messages_to_budget_prunes_history_and_context(self):
        tok = ChatTemplateTokenizer()
        long_context = '[S1] ' + ('data ' * 500)
        history = [
            {'role': 'user', 'content': 'q1 ' + ('a ' * 50)},
            {'role': 'assistant', 'content': 'ans1 ' + ('b ' * 50)},
            {'role': 'user', 'content': 'q2 ' + ('c ' * 50)},
            {'role': 'assistant', 'content': 'ans2 ' + ('d ' * 50)},
        ]
        user_msg = 'pertanyaan panjang ' + ('kata ' * 100)

        bare_messages = [
            {'role': 'system', 'content': make_system('terstruktur', '')},
            {'role': 'user', 'content': user_msg},
        ]
        bare_tokens = count_prompt_tokens(bare_messages, tok)

        tight_budget = bare_tokens + 150
        messages, tokens = fit_messages_to_budget(
            mode='terstruktur',
            context=long_context,
            user_message=user_msg,
            history=history,
            tokenizer=tok,
            max_prompt_tokens=tight_budget,
        )
        self.assertLessEqual(tokens, tight_budget)
        self.assertNotIn('ans1', [m['content'] for m in messages])

    def test_resolve_max_prompt_tokens_auto_maximum(self):
        self.assertEqual(MODEL_MAX_CONTEXT_TOKENS, 32768)
        self.assertEqual(resolve_max_prompt_tokens(), 32224)
        self.assertEqual(resolve_max_prompt_tokens(configured=0), 32224)

        class DummyConfig:
            max_position_embeddings = 16384

        class DummyModel:
            config = DummyConfig()

        self.assertEqual(resolve_max_prompt_tokens(DummyModel(), max_new_tokens=512), 15840)
        self.assertEqual(resolve_max_prompt_tokens(DummyModel(), configured=4096), 4096)
        self.assertEqual(resolve_max_prompt_tokens(DummyModel(), configured=99999), 15840)

    def test_fit_messages_to_budget_middle_truncates_ultra_long_user_prompt(self):
        tok = ChatTemplateTokenizer()
        empty_sys_tokens = count_prompt_tokens(
            [{'role': 'system', 'content': make_system('terstruktur', '')}, {'role': 'user', 'content': ''}],
            tok,
        )
        budget = empty_sys_tokens + 400
        ultra_long_prompt = (
            'PEMBUKA_KONTEKS_SISWA '
            + ('isi tengah sangat panjang ' * 80)
            + 'PERTANYAAN_INTI_AKHIR?'
        )
        messages, tokens = fit_messages_to_budget(
            mode='terstruktur',
            context='[S1] Materi RAG',
            user_message=ultra_long_prompt,
            tokenizer=tok,
            max_prompt_tokens=budget,
        )
        self.assertLessEqual(tokens, budget)
        user_content = messages[-1]['content']
        self.assertIn('PEMBUKA_KONTEKS_SISWA', user_content)
        self.assertIn('PERTANYAAN_INTI_AKHIR?', user_content)
        self.assertIn('dipersingkat otomatis', user_content)

    def test_fit_messages_to_budget_raises_when_prompt_exceeds_entire_budget(self):
        tok = ChatTemplateTokenizer()
        with self.assertRaises(ValueError) as ctx:
            fit_messages_to_budget(
                mode='terstruktur',
                context='',
                user_message='teks ' * 100,
                tokenizer=tok,
                max_prompt_tokens=50,
            )
        self.assertIn('Pertanyaan terlalu panjang', str(ctx.exception))
        self.assertIn('batas: 50', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
