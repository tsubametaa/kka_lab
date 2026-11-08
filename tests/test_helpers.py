import unittest
from app.features.chat import (
    DEFAULT_MAX_PROMPT_TOKENS,
    MODEL_MAX_CONTEXT_TOKENS,
    count_prompt_tokens,
    fit_messages_to_budget,
    make_system,
    resolve_max_prompt_tokens,
)
from app.features.dataset import (
    encode_target,
    split_groups,
    validate_rows,
)
from app.features.rag import (
    resolve_query_topic,
    retrieval_query,
)

class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = ''.join(f'<{m["role"]}>{m["content"]}</end>' for m in messages)
        if add_generation_prompt:
            rendered += '<assistant>'
        return [ord(x) for x in rendered]

class Helpers(unittest.TestCase):
    def test_grouped_split(self):
        rows = [{'group_id': str(i), 'mode': mode} for i in range(12)
                for mode in ('umum', 'terstruktur')]
        train, val = split_groups(rows)
        self.assertFalse({r['group_id'] for r in train} & {r['group_id'] for r in val})
        for part in (train, val):
            for group in {r['group_id'] for r in part}:
                self.assertEqual({r['mode'] for r in part if r['group_id'] == group},
                                 {'umum', 'terstruktur'})

    def test_target_mask(self):
        tok = FakeTokenizer()
        messages = [{'role': 'user', 'content': 'apa'},
                    {'role': 'assistant', 'content': 'jawab'}]
        item = encode_target(tok, messages, 1000)
        prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True)
        self.assertTrue(all(x == -100 for x in item['labels'][:len(prefix)]))
        self.assertEqual(item['labels'][len(prefix):], item['input_ids'][len(prefix):])
        with self.assertRaises(ValueError):
            encode_target(tok, messages, 5)

    def test_validation_and_modes(self):
        base = {'group_id': 'a', 'context': '[S1] materi',
                'messages': [{'role': 'user', 'content': 'apa'},
                             {'role': 'assistant', 'content':
                              '1. Periksa materi. 2. Bandingkan hasil karena tujuan berbeda. '
                              'Menurutmu apa alasannya?'}]}
        rows = [dict(base, id=mode, mode=mode) for mode in ('umum', 'terstruktur')]
        validate_rows(rows)
        raw_answer_rows = [
            dict(row, messages=[row['messages'][0], {'role': 'assistant', 'content': 'Jawab langsung.'}])
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, 'minimal dua langkah'):
            validate_rows(raw_answer_rows)
        no_reason_rows = [
            dict(row, messages=[row['messages'][0], {'role': 'assistant',
                 'content': '1. Amati contoh. 2. Bandingkan hasil. Menurutmu apa bedanya?'}])
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, 'menjelaskan alasan'):
            validate_rows(no_reason_rows)
        no_prompt_rows = [
            dict(row, messages=[row['messages'][0], {'role': 'assistant',
                 'content': '1. Amati contoh. 2. Bandingkan hasil karena tujuannya berbeda.'}])
            for row in rows
        ]
        with self.assertRaisesRegex(ValueError, 'pemantik berpikir'):
            validate_rows(no_prompt_rows)
        with self.assertRaises(ValueError):
            validate_rows(rows + [rows[0]])
        general = make_system('umum', 'x')
        structured = make_system('terstruktur', 'x')
        self.assertNotEqual(general, structured)
        self.assertIn('Jangan memberikan jawaban akhir saja', general)
        self.assertIn('2-4 langkah', general)
        self.assertIn('Jangan berikan solusi lengkap pada giliran awal', structured)
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
        self.assertIn('SOURCE CONTEXT', general_en)
        self.assertIn('PERSONAL MEMORY', general_en)
        self.assertIn('MANDATORY LANGUAGE RULE', general_en)
        self.assertIn('Never answer in Indonesian.', general_en)
        with self.assertRaises(ValueError):
            make_system('umum', 'x', lang='fr')

    def test_retrieval_query_and_topic_resolution(self):
        # Short question (< 6 words) with topic should merge
        merged = retrieval_query('contohnya?', 'apa itu data latih?')
        self.assertEqual(merged, 'apa itu data latih? contohnya?')

        query, next_topic = resolve_query_topic('contohnya?', 'apa itu data latih?')
        self.assertEqual(query, 'apa itu data latih? contohnya?')
        self.assertEqual(next_topic, 'apa itu data latih?')

        # Long question (>= 6 words) should not merge and should update topic
        long_q = 'bagaimana cara membagi data latih dan data validasi secara tepat?'
        self.assertEqual(retrieval_query(long_q, 'topik lama'), long_q)
        q2, t2 = resolve_query_topic(long_q, 'topik lama')
        self.assertEqual(q2, long_q)
        self.assertEqual(t2, long_q)

        # Empty topic should not merge
        self.assertEqual(retrieval_query('apa itu?', ''), 'apa itu?')

    def test_fit_messages_to_budget_normal(self):
        tok = FakeTokenizer()
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
        tok = FakeTokenizer()
        long_context = '[S1] ' + ('data ' * 500)
        history = [
            {'role': 'user', 'content': 'q1 ' + ('a ' * 50)},
            {'role': 'assistant', 'content': 'ans1 ' + ('b ' * 50)},
            {'role': 'user', 'content': 'q2 ' + ('c ' * 50)},
            {'role': 'assistant', 'content': 'ans2 ' + ('d ' * 50)},
        ]
        user_msg = 'pertanyaan panjang ' + ('kata ' * 100)

        # Baseline cost with zero history and zero context
        bare_messages = [
            {'role': 'system', 'content': make_system('terstruktur', '')},
            {'role': 'user', 'content': user_msg},
        ]
        bare_tokens = count_prompt_tokens(bare_messages, tok)

        # Budget large enough for bare messages + partial context, but too small for full history + full context
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
        # History should have been pruned to save budget for the long prompt
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
        # Explicit smaller override is respected, while oversized override clamps to arch_max
        self.assertEqual(resolve_max_prompt_tokens(DummyModel(), configured=4096), 4096)
        self.assertEqual(resolve_max_prompt_tokens(DummyModel(), configured=99999), 15840)

    def test_fit_messages_to_budget_middle_truncates_ultra_long_user_prompt(self):
        tok = FakeTokenizer()
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
        tok = FakeTokenizer()
        with self.assertRaises(ValueError) as ctx:
            fit_messages_to_budget(
                mode='terstruktur',
                context='',
                user_message='teks ' * 100,
                tokenizer=tok,
                max_prompt_tokens=50,  # Ridiculously small budget
            )
        self.assertIn('Pertanyaan terlalu panjang', str(ctx.exception))
        self.assertIn('batas: 50', str(ctx.exception))

if __name__ == '__main__':
    unittest.main()
