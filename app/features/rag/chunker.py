"""Markdown-aware chunker preserving headings, tables, formulas, and PDF page numbers."""

import re
from dataclasses import dataclass

DEFAULT_CHUNK_TOKENS = 220
DEFAULT_OVERLAP_TOKENS = 35
MAX_EMBED_TOKENS = 320
MAX_CHUNK_CHARS = 1100

_HEADING_RE = re.compile(r'^(#{1,4})\s+(.+?)\s*$')
_FENCE_RE = re.compile(r'^(\s*(?:```|~~~|\$\$))')
_TABLE_LINE_RE = re.compile(r'^\s*\|.*\|\s*$')
_EMBED_PART_RE = re.compile(r'[A-Za-zÀ-ÿ]+|\d|[^\w\s]')


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    text: str
    topic: str
    pages_pdf: int | list[int]


def _estimate_embed_tokens(text):
    """Conservative upper-bound token estimate for WordPiece/E5 512-token embedding APIs."""
    if not text:
        return 0
    total = 0
    for part in _EMBED_PART_RE.findall(text):
        if len(part) <= 4:
            total += 1
        else:
            total += (len(part) + 3) // 4
    return max(1, total)


def _count_tokens(text, tokenizer=None):
    if not text:
        return 0
    est = _estimate_embed_tokens(text)
    if tokenizer is not None:
        return max(len(tokenizer.encode(text, add_special_tokens=False)), est)
    return est


def _parse_markdown_blocks(pages):
    """Ubah daftar (page_number, markdown) menjadi blok atomik beserta jalur heading aktif."""
    headings = {}
    blocks = []

    for item in pages:
        if isinstance(item, dict):
            page_num = int(item['page_number'])
            md = str(item.get('markdown', ''))
        else:
            page_num = int(item.page_number)
            md = str(item.markdown)

        lines = md.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        idx = 0
        total = len(lines)

        while idx < total:
            line = lines[idx]
            stripped = line.strip()
            if not stripped:
                idx += 1
                continue

            heading_match = _HEADING_RE.match(stripped)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                headings[level] = title
                for deeper in [k for k in headings if k > level]:
                    del headings[deeper]
                idx += 1
                continue

            section_parts = [headings[k] for k in sorted(headings)]

            # Blok kode atau rumus blok (``` ... ``` atau $$ ... $$) dijaga tetap utuh.
            fence_match = _FENCE_RE.match(line)
            if fence_match:
                marker = fence_match.group(1).strip()[:3] if not stripped.startswith('$$') else '$$'
                buf = [line]
                idx += 1
                if marker == '$$' and stripped.endswith('$$') and len(stripped) > 4:
                    pass
                else:
                    while idx < total:
                        buf.append(lines[idx])
                        cur = lines[idx].strip()
                        idx += 1
                        if cur.startswith(marker):
                            break
                blocks.append((page_num, list(section_parts), '\n'.join(buf).strip()))
                continue

            # Tabel Markdown (baris berturut-turut diawali & diakhiri '|') dijaga tetap utuh.
            if _TABLE_LINE_RE.match(line):
                buf = [line]
                idx += 1
                while idx < total and _TABLE_LINE_RE.match(lines[idx]):
                    buf.append(lines[idx])
                    idx += 1
                blocks.append((page_num, list(section_parts), '\n'.join(buf).strip()))
                continue

            # Paragraf atau daftar biasa sampai baris kosong / heading / tabel / fence berikutnya.
            buf = [line]
            idx += 1
            while idx < total:
                nxt = lines[idx]
                if not nxt.strip():
                    break
                if _HEADING_RE.match(nxt.strip()) or _FENCE_RE.match(nxt) or _TABLE_LINE_RE.match(nxt):
                    break
                buf.append(nxt)
                idx += 1
            blocks.append((page_num, list(section_parts), '\n'.join(buf).strip()))

    return blocks


def _split_long_unit(unit, max_tokens, tokenizer, max_chars=MAX_CHUNK_CHARS):
    """Split a single long line/sentence by words, then hard slice unbroken tokens."""
    if _count_tokens(unit, tokenizer) <= max_tokens and len(unit) <= max_chars:
        return [unit]
    pieces, cur = [], []
    for word in unit.split():
        if _count_tokens(word, tokenizer) > max_tokens or len(word) > max_chars:
            if cur:
                pieces.append(' '.join(cur))
                cur = []
            step = max(80, min(max_chars // 2, max_tokens * 2))
            for i in range(0, len(word), step):
                pieces.append(word[i:i + step])
            continue
        candidate = ' '.join([*cur, word]) if cur else word
        if cur and (_count_tokens(candidate, tokenizer) > max_tokens or len(candidate) > max_chars):
            pieces.append(' '.join(cur))
            cur = [word]
        else:
            cur.append(word)
    if cur:
        pieces.append(' '.join(cur))
    return pieces or [unit[:max_chars]]


def _split_oversized_block(text, max_tokens, tokenizer, max_chars=MAX_CHUNK_CHARS):
    """Pecah satu blok yang terlalu panjang berdasarkan baris/kalimat/kata."""
    if _count_tokens(text, tokenizer) <= max_tokens and len(text) <= max_chars:
        return [text]

    lines = text.split('\n')
    # Jika tabel panjang, pertahankan header tabel (2 baris pertama) di tiap pecahan.
    if len(lines) >= 3 and all(_TABLE_LINE_RE.match(ln) for ln in lines[:2]):
        table_header = '\n'.join(lines[:2])
        header_tokens = _count_tokens(table_header, tokenizer) + 2
        if header_tokens < max_tokens and len(table_header) < max_chars:
            pieces, current, cur_tokens = [], [table_header], header_tokens
            cur_chars = len(table_header)
            for row in lines[2:]:
                row_tokens = _count_tokens(row, tokenizer) + 1
                if _count_tokens(row, tokenizer) + header_tokens > max_tokens or len(row) + len(table_header) > max_chars:
                    if len(current) > 1:
                        pieces.append('\n'.join(current))
                        current, cur_tokens, cur_chars = [table_header], header_tokens, len(table_header)
                    pieces.extend(_split_long_unit(row, max_tokens, tokenizer, max_chars=max_chars))
                    continue
                if len(current) > 1 and (cur_tokens + row_tokens > max_tokens or cur_chars + len(row) + 1 > max_chars):
                    pieces.append('\n'.join(current))
                    current = [table_header, row]
                    cur_tokens = header_tokens + row_tokens
                    cur_chars = len(table_header) + len(row) + 1
                else:
                    current.append(row)
                    cur_tokens += row_tokens
                    cur_chars += len(row) + 1
            if len(current) > 1:
                pieces.append('\n'.join(current))
            if pieces:
                return pieces

    units = re.split(r'(?<=[.!?])\s+|\n+', text)
    pieces, current, cur_tokens, cur_chars = [], [], 0, 0
    for raw_unit in units:
        raw_unit = raw_unit.strip()
        if not raw_unit:
            continue
        for unit in _split_long_unit(raw_unit, max_tokens, tokenizer, max_chars=max_chars):
            u_tokens = _count_tokens(unit, tokenizer) + 1
            if current and (cur_tokens + u_tokens > max_tokens or cur_chars + len(unit) + 1 > max_chars):
                pieces.append(' '.join(current))
                current, cur_tokens, cur_chars = [unit], u_tokens, len(unit)
            else:
                current.append(unit)
                cur_tokens += u_tokens
                cur_chars += len(unit) + 1
    if current:
        pieces.append(' '.join(current))
    return pieces or _split_long_unit(text, max_tokens, tokenizer, max_chars=max_chars)

def _format_pages(page_numbers):
    unique = sorted(set(page_numbers))
    if len(unique) == 1:
        return unique[0]
    return unique

def chunk_markdown_pages(pages, *, max_tokens=DEFAULT_CHUNK_TOKENS,
                         overlap_tokens=DEFAULT_OVERLAP_TOKENS,
                         tokenizer=None, default_topic=''):
    """Pecah hasil Markdown per halaman menjadi daftar DocumentChunk siap Astra DB."""
    if max_tokens < 48:
        raise ValueError('max_tokens minimal 48.')
    overlap_tokens = max(0, min(overlap_tokens, max_tokens // 2))

    raw_blocks = _parse_markdown_blocks(pages)
    if not raw_blocks:
        return []

    expanded = []
    for page_num, section_parts, body in raw_blocks:
        section_header = ' > '.join(section_parts)
        prefix_tokens = _count_tokens(f'[{section_header}]\n', tokenizer) if section_header else 0
        body_budget = max(32, max_tokens - prefix_tokens)
        for piece in _split_oversized_block(body, body_budget, tokenizer):
            expanded.append((page_num, section_parts, piece))

    chunks = []
    buf_items = []  # list of (page_num, piece_text, piece_tokens)
    buf_section = None
    buf_tokens = 0

    def flush():
        nonlocal buf_items, buf_tokens
        if not buf_items:
            return
        section_str = ' > '.join(buf_section) if buf_section else ''
        body_text = '\n\n'.join(item[1] for item in buf_items).strip()
        if section_str:
            full_text = f'[{section_str}]\n{body_text}'
            topic = buf_section[-1]
        else:
            full_text = body_text
            topic = default_topic
        pages_val = _format_pages(item[0] for item in buf_items)
        chunks.append(DocumentChunk(
            index=len(chunks),
            text=full_text,
            topic=topic or default_topic,
            pages_pdf=pages_val,
        ))

    for page_num, section_parts, piece in expanded:
        p_tokens = _count_tokens(piece, tokenizer) + 2
        section_tuple = tuple(section_parts)
        header_cost = _count_tokens(f"[{' > '.join(section_tuple)}]\n", tokenizer) if section_tuple else 0

        section_changed = buf_section is not None and section_tuple != buf_section
        exceeds_budget = buf_items and (header_cost + buf_tokens + p_tokens > max_tokens)

        if section_changed or exceeds_budget:
            prev_items = list(buf_items)
            prev_section = buf_section
            flush()
            buf_items = []
            buf_tokens = 0
            # Overlap hanya dipertahankan jika masih dalam sub-bab (heading) yang sama.
            if not section_changed and overlap_tokens > 0 and prev_section == section_tuple:
                carried, carried_tokens = [], 0
                for old_item in reversed(prev_items):
                    if carried_tokens + old_item[2] > overlap_tokens:
                        break
                    carried.append(old_item)
                    carried_tokens += old_item[2]
                buf_items = list(reversed(carried))
                buf_tokens = carried_tokens

        buf_section = section_tuple
        buf_items.append((page_num, piece, p_tokens))
        buf_tokens += p_tokens

    flush()
    return chunks
