"""Dataset grouping and tokenizer encoding for LoRA training."""

import random


def split_groups(rows, seed=42, fraction=0.15):
    groups = sorted({r['group_id'] for r in rows})
    if len(groups) < 5:
        raise ValueError('Minimal 5 kelompok skenario untuk uji alur split.')
    random.Random(seed).shuffle(groups)
    nval = max(1, round(len(groups) * fraction))
    val_ids = set(groups[:nval])
    train = [r for r in rows if r['group_id'] not in val_ids]
    val = [r for r in rows if r['group_id'] in val_ids]
    return train, val


def encode_target(tokenizer, messages, max_length):
    if messages[-1]['role'] != 'assistant':
        raise ValueError('Pesan terakhir harus target assistant.')
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if full[:len(prefix)] != prefix:
        raise ValueError('Batas chat template tidak cocok; periksa tokenizer.')
    if len(full) > max_length:
        raise ValueError(f'Contoh {len(full)} token melebihi {max_length}.')
    if len(full) <= len(prefix):
        raise ValueError('Tidak ada token jawaban untuk dilatih.')
    return {
        'input_ids': full,
        'attention_mask': [1] * len(full),
        'labels': [-100] * len(prefix) + full[len(prefix):],
    }

