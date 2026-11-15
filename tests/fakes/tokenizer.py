"""Lightweight tokenizer test doubles for prompt budgeting and dataset tests."""


class ChatTemplateTokenizer:
    """Deterministic character-level tokenizer implementing apply_chat_template."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        rendered = ''.join(f'<{m["role"]}>{m["content"]}</end>' for m in messages)
        if add_generation_prompt:
            rendered += '<assistant>'
        return [ord(x) for x in rendered]


class WordTokenizer:
    """Minimal whitespace word tokenizer supporting encode and decode."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(ids)
