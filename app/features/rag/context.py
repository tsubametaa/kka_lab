"""RAG context builder and conversational topic resolution."""

SHORT_QUESTION_WORDS = 6

def retrieval_query(question, topic=""):
    """Pertanyaan lanjutan pendek (<6 kata) digabung dengan topik sebelumnya."""
    if topic and len(question.split()) < SHORT_QUESTION_WORDS:
        return f"{topic} {question}"
    return question

def resolve_query_topic(question, topic=""):
    """Mengembalikan (query_rag, topik_berikutnya) untuk percakapan interaktif."""
    query = retrieval_query(question, topic)
    next_topic = topic if query != question else question
    return query, next_topic

def build_context(docs, tokenizer, max_tokens=900):
    blocks, sources = [], []
    used = 0
    for doc in docs:
        meta = doc.get('metadata') or {}
        text = str(doc.get('text', '')).strip()
        if not text:
            continue
        title = str(doc.get('source_title') or meta.get('source_title') or 'Materi KKA')[:140]
        tag = f'S{len(sources) + 1}'
        header = f'[{tag}] {title}\n'
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        remaining = max_tokens - used - len(header_ids) - 4
        if remaining < 24:
            break
        text_ids = tokenizer.encode(text, add_special_tokens=False)[:remaining]
        body = tokenizer.decode(text_ids, skip_special_tokens=True)
        blocks.append(header + body)
        used += len(header_ids) + len(text_ids) + 4
        sources.append({
            'label': tag, 'id': str(doc.get('_id', '')),
            'title': title, 'similarity': doc.get('$similarity'),
            'pages_pdf': doc.get('pages_pdf') or meta.get('pages_pdf'),
        })
    return '\n\n'.join(blocks), sources