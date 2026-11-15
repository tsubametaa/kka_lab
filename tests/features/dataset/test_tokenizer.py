"""Tests for app.features.dataset.tokenizer (group splits and label masking)."""

import unittest

from app.features.dataset import encode_target, split_groups
from tests.fakes import ChatTemplateTokenizer


class DatasetTokenizerTests(unittest.TestCase):
    def test_grouped_split(self):
        rows = [
            {'group_id': str(i), 'mode': mode}
            for i in range(12)
            for mode in ('umum', 'terstruktur')
        ]
        train, val = split_groups(rows)
        self.assertFalse({r['group_id'] for r in train} & {r['group_id'] for r in val})
        for part in (train, val):
            for group in {r['group_id'] for r in part}:
                self.assertEqual(
                    {r['mode'] for r in part if r['group_id'] == group},
                    {'umum', 'terstruktur'},
                )

    def test_target_mask(self):
        tok = ChatTemplateTokenizer()
        messages = [
            {'role': 'user', 'content': 'apa'},
            {'role': 'assistant', 'content': 'jawab'},
        ]
        item = encode_target(tok, messages, 1000)
        prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True)
        self.assertTrue(all(x == -100 for x in item['labels'][:len(prefix)]))
        self.assertEqual(item['labels'][len(prefix):], item['input_ids'][len(prefix):])
        with self.assertRaises(ValueError):
            encode_target(tok, messages, 5)


if __name__ == '__main__':
    unittest.main()
