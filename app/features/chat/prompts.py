"""System prompts and pedagogical scaffolding definitions."""

from app.features.language.locales import LOCALES, id as id_locale

BASE = id_locale.BASE
PEDAGOGY = id_locale.PEDAGOGY
FORMAT = id_locale.FORMAT
POLICIES = id_locale.POLICIES

MODEL_MAX_CONTEXT_TOKENS = 32768
DEFAULT_MAX_PROMPT_TOKENS = MODEL_MAX_CONTEXT_TOKENS - 512 - 32  # 32224
DEFAULT_MAX_NEW_TOKENS = 768
DEFAULT_SAMPLING_CONFIG = {
    'do_sample': True,
    'temperature': 0.45,
    'top_p': 0.9,
    'min_p': 0.05,
    'repetition_penalty': 1.08,
}

def build_generation_kwargs(
    tokenizer,
    *,
    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    do_sample=DEFAULT_SAMPLING_CONFIG['do_sample'],
    temperature=DEFAULT_SAMPLING_CONFIG['temperature'],
    top_p=DEFAULT_SAMPLING_CONFIG['top_p'],
    min_p=DEFAULT_SAMPLING_CONFIG['min_p'],
    repetition_penalty=DEFAULT_SAMPLING_CONFIG['repetition_penalty'],
):
    """Build Hugging Face generate() kwargs with safe natural sampling defaults."""
    kwargs = {
        'max_new_tokens': max(1, int(max_new_tokens)),
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    if do_sample:
        temp = float(temperature)
        if temp <= 0:
            raise ValueError('temperature harus lebih besar dari 0 saat do_sample=True.')
        kwargs.update({
            'do_sample': True,
            'temperature': temp,
            'top_p': float(top_p),
            'min_p': float(min_p),
            'repetition_penalty': float(repetition_penalty),
        })
    else:
        kwargs.update({
            'do_sample': False,
            'temperature': None,
            'top_p': None,
            'top_k': None,
        })
        if repetition_penalty and float(repetition_penalty) != 1.0:
            kwargs['repetition_penalty'] = float(repetition_penalty)
    return kwargs


def resolve_max_prompt_tokens(model=None, max_new_tokens=512, configured=None):
    """Return the highest safe prompt token limit supported by the model architecture."""
    config = getattr(model, 'config', None) or getattr(
        getattr(model, 'base_model', None), 'config', None
    )
    ctx_limit = getattr(config, 'max_position_embeddings', None)
    if not isinstance(ctx_limit, int) or ctx_limit <= 0:
        ctx_limit = MODEL_MAX_CONTEXT_TOKENS
    arch_max = max(1024, ctx_limit - max(1, int(max_new_tokens)) - 32)
    if configured is not None and int(configured) > 0:
        return min(int(configured), arch_max)
    return arch_max

def make_system(mode, context, memory='', lang='id', persona=''):
    if lang not in LOCALES:
        raise ValueError('Bahasa harus id atau en.')
    locale = LOCALES[lang]
    if mode not in locale.POLICIES:
        raise ValueError('Mode harus umum atau terstruktur.')
    source = context.strip() or locale.EMPTY_CONTEXT
    system = (
        locale.BASE + '\n' + locale.PEDAGOGY + '\n' + locale.FORMAT + '\n'
        + locale.POLICIES[mode] + '\n'
    )
    if persona.strip():
        # Persona preferensi gaya dari siswa
        system += (
            locale.PERSONA_HEADER + '\n<persona_siswa>\n' + persona.strip()
            + '\n</persona_siswa>\n' + locale.PERSONA_RULES + '\n'
        )
    system += locale.CONTEXT_HEADER + '\n' + source
    if memory.strip():
        system += (
            '\n' + locale.MEMORY_HEADER + '\n'
            + memory.strip()
            + '\n' + locale.MEMORY_FOOTER
        )
    system += '\n\n' + locale.LANGUAGE_FOOTER
    return system


def count_prompt_tokens(messages, tokenizer):
    """Estimate or count total tokens for chat messages using tokenizer."""
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            tokens = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            if hasattr(tokens, 'shape'):
                return int(tokens.shape[-1])
            if isinstance(tokens, (list, tuple)):
                return len(tokens)
        except Exception:
            pass
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if callable(tokenizer):
                encoded = tokenizer(rendered, add_special_tokens=False)
                if isinstance(encoded, dict) and 'input_ids' in encoded:
                    ids = encoded['input_ids']
                    return ids.shape[-1] if hasattr(ids, 'shape') else len(ids)
                if hasattr(encoded, 'shape'):
                    return int(encoded.shape[-1])
                if isinstance(encoded, (list, tuple)):
                    return len(encoded)
            return len(rendered.split())
        except Exception:
            pass
    return sum(len(m.get('content', '').split()) for m in messages)


def _truncate_text_to_tokens(text, max_tokens, tokenizer):
    """Safely truncate text to fit within max_tokens."""
    if not text or max_tokens <= 0:
        return ''
    if tokenizer is not None and hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True).strip()
        except Exception:
            pass
    words = text.split()
    max_words = max(1, int(max_tokens * 0.65))
    return ' '.join(words[:max_words])


def _truncate_user_middle(text, max_tokens, tokenizer, lang='id'):
    """Preserve both the opening context and closing question of an ultra-long prompt."""
    if not text or max_tokens <= 0:
        return ''
    if max_tokens < 200:
        marker = '\n[...]\n'
    elif lang == 'en':
        marker = '\n\n[... middle section condensed to fit model context ...]\n\n'
    else:
        marker = '\n\n[... bagian tengah dipersingkat otomatis agar muat dalam konteks model ...]\n\n'
    if tokenizer is not None and hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            usable = max(16, max_tokens - len(marker_ids))
            head_len = max(8, int(usable * 0.4))
            tail_len = max(8, usable - head_len)
            head = tokenizer.decode(ids[:head_len], skip_special_tokens=True).strip()
            tail = tokenizer.decode(ids[-tail_len:], skip_special_tokens=True).strip()
            return f'{head}{marker}{tail}'
        except Exception:
            pass
    # Character/word fallback when tokenizer encode/decode is unavailable (e.g. FakeTokenizer)
    max_chars = max(24, max_tokens - len(marker))
    if len(text) <= max_chars:
        return text
    head_chars = max(10, int(max_chars * 0.4))
    tail_chars = max(10, max_chars - head_chars)
    return f'{text[:head_chars].rstrip()}{marker}{text[-tail_chars:].lstrip()}'


def fit_messages_to_budget(
    mode,
    context,
    user_message,
    history=None,
    tokenizer=None,
    max_prompt_tokens=DEFAULT_MAX_PROMPT_TOKENS,
    memory='',
    lang='id',
    persona='',
):
    """Progressively prune history, memory, RAG context, and middle of ultra-long prompts.

    Stages:
    1. Full system prompt (with context & memory) + full history + user message.
    2. Prune history (oldest turns first).
    3. Drop personal memory if still over budget.
    4. Adaptively truncate RAG context to fit available headroom.
    5. Drop RAG context entirely.
    6. Middle-truncate ultra-long user messages (>32K tokens) keeping opening + closing.
    7. Raise ValueError only if max_prompt_tokens cannot even hold the base system prompt.
    """
    recent = list(history or [])
    current_memory = memory
    current_context = context

    def _build(ctx, mem, hist, user_text=user_message):
        sys_content = make_system(mode, ctx, memory=mem, lang=lang, persona=persona)
        return [
            {'role': 'system', 'content': sys_content},
            *hist,
            {'role': 'user', 'content': user_text},
        ]

    # Stage 1: Full system + RAG context + memory + full history
    messages = _build(current_context, current_memory, recent)
    tokens = count_prompt_tokens(messages, tokenizer)
    if tokens <= max_prompt_tokens:
        return messages, tokens

    # Stage 2: Prune conversation history (oldest first)
    while recent and tokens > max_prompt_tokens:
        recent = recent[2:]
        messages = _build(current_context, current_memory, recent)
        tokens = count_prompt_tokens(messages, tokenizer)
        if tokens <= max_prompt_tokens:
            return messages, tokens

    # Stage 3: Drop personal memory if still over budget
    if current_memory and tokens > max_prompt_tokens:
        current_memory = ''
        messages = _build(current_context, current_memory, recent)
        tokens = count_prompt_tokens(messages, tokenizer)
        if tokens <= max_prompt_tokens:
            return messages, tokens

    # Stage 4: Adaptively prune RAG context so long prompts still fit
    if current_context and tokens > max_prompt_tokens:
        baseline_messages = _build('', '', recent)
        baseline_tokens = count_prompt_tokens(baseline_messages, tokenizer)
        if baseline_tokens <= max_prompt_tokens:
            avail_tokens = max_prompt_tokens - baseline_tokens - 16
            if avail_tokens >= 50:
                trimmed_context = _truncate_text_to_tokens(
                    current_context, avail_tokens, tokenizer
                )
                messages = _build(trimmed_context, '', recent)
                tokens = count_prompt_tokens(messages, tokenizer)
                if tokens <= max_prompt_tokens:
                    return messages, tokens
            # Headroom is too small for meaningful context, use empty context
            return baseline_messages, baseline_tokens

    # Stage 5: Zero history, zero memory, zero context
    minimal_messages = _build('', '', [])
    final_tokens = count_prompt_tokens(minimal_messages, tokenizer)
    if final_tokens <= max_prompt_tokens:
        return minimal_messages, final_tokens

    # Stage 6: Middle-truncate ultra-long user prompt if base system prompt leaves room
    empty_tokens = count_prompt_tokens(_build('', '', [], ''), tokenizer)
    avail_user_tokens = max_prompt_tokens - empty_tokens - 16
    if avail_user_tokens >= 64:
        budget_for_user = avail_user_tokens
        while budget_for_user >= 32:
            shortened_user = _truncate_user_middle(
                user_message, budget_for_user, tokenizer, lang=lang
            )
            candidate = _build('', '', [], shortened_user)
            candidate_tokens = count_prompt_tokens(candidate, tokenizer)
            if candidate_tokens <= max_prompt_tokens:
                return candidate, candidate_tokens
            budget_for_user = int(budget_for_user * 0.85) - 8

    raise ValueError(
        f'Pertanyaan terlalu panjang ({final_tokens} token, batas: {max_prompt_tokens}). '
        f'Ringkas pertanyaan atau gunakan opsi --max-prompt-tokens yang lebih besar.'
    )


