"""Markdown-aware chunker preserving headings, tables, formulas, and PDF page numbers."""

import re
from dataclasses import dataclass

DEFAULT_CHUNK_TOKENS = 300
DEFAULT_OVERLAP_TOKENS = 40

_HEADING_RE = re.compile(r'^(#{1,4})\s+(.+?)\s*$')
_FENCE_RE = re.compile(r'^(\s*(?:```|~~~|\$\$))')
_TABLE_LINE_RE = re.compile(r'^\s*\|.*\|\s*$')


@dataclass(frozen=True)
class DocumentChunk:
    index: int
    text: str
    topic: str
    pages_pdf: int | list[int]


def _count_tokens(text, tokenizer=None):
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    # Estimasi konservatif ~3.5 karakter per token untuk teks Indonesia/Inggris + LaTeX.
    return max(1, (len(text) + 2) // 3)


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


def _split_oversized_block(text, max_tokens, tokenizer):
    """Pecah satu blok yang terlalu panjang berdasarkan baris/kalimat."""
    if _count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    lines = text.split('\n')
    # Jika tabel panjang, pertahankan header tabel (2 baris pertama) di tiap pecahan.
    if len(lines) >= 3 and all(_TABLE_LINE_RE.match(ln) for ln in lines[:2]):
        table_header = '\n'.join(lines[:2])
        header_tokens = _count_tokens(table_header, tokenizer) + 2
        pieces, current, cur_tokens = [], [table_header], header_tokens
        for row in lines[2:]:
            row_tokens = _count_tokens(row, tokenizer) + 1
            if len(current) > 1 and cur_tokens + row_tokens > max_tokens:
                pieces.append('\n'.join(current))
                current, cur_tokens = [table_header, row], header_tokens + row_tokens
            else:
                current.append(row)
                cur_tokens += row_tokens
        if len(current) > 1:
            pieces.append('\n'.join(current))
        return pieces or [text]

    units = re.split(r'(?<=[.!?])\s+|\n+', text)
    pieces, current, cur_tokens = [], [], 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        u_tokens = _count_tokens(unit, tokenizer) + 1
        if current and cur_tokens + u_tokens > max_tokens:
            pieces.append(' '.join(current))
            current, cur_tokens = [unit], u_tokens
        else:
            current.append(unit)
            cur_tokens += u_tokens
    if current:
        pieces.append(' '.join(current))
    return pieces or [text]

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
