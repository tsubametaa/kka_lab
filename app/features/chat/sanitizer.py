"""Response artifact sanitization for non-streamed and streamed chat outputs."""

import re

_CODE_SEGMENT_RE = re.compile(r'(```[\s\S]*?```|`[^`\n]+`)')
_CITATION_TAG_RE = re.compile(r'(?<!\w)\[\s*(?:S|M)\d+(?:\s*[,;]\s*(?:S|M)\d+)*\s*\](?!\s*=)')
_SPECIAL_TOKEN_RE = re.compile(
    r'<\|(?:im_end|im_start|endoftext|eot_id|begin_of_text|end_of_text)[^|>]*\|?>'
    r'|<｜(?:end▁of▁sentence|begin▁of▁sentence)｜>'
    r'|<\|(?:im_end|im_start|endoftext)$'
)
_XML_WRAPPER_RE = re.compile(
    r'</?(?:konteks_sumber|memori_personal|persona_siswa|nama_siswa)>',
    re.IGNORECASE,
)
_META_LINE_RE = re.compile(
    r'^[ \t]*(?:Topik|Konteks Sumber|Memori Personal|Sumber Referensi):\s*.*(?:\n|$)',
    re.IGNORECASE | re.MULTILINE,
)
_RAW_ID_SCORE_RE = re.compile(
    r'\s*\(?\b(?:doc_id|id)\s*:\s*auto_kka_[0-9a-f]+\b\)?'
    r'|\bauto_kka_[0-9a-f]{6,}\b'
    r'|\s*\(?\b(?:skor|similarity)\s*[:=]?\s*0\.\d+\b\)?',
    re.IGNORECASE,
)
_SPACE_BEFORE_PUNCT_RE = re.compile(r'[ \t]+([.,!?;:])')
_MULTI_SPACE_RE = re.compile(r' {2,}')
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')

_PARTIAL_TAIL_RE = re.compile(
    r'(?:\s?\[(?:(?:S|M)\d*(?:\s*[,;]\s*(?:S|M)?\d*)*)?'
    r'|<(?:(?:\||｜)[^>\n]{0,32}'
    r'|/?(?:konteks_sumber|memori_personal|persona_siswa|nama_siswa)?'
    r'|k|m|p|n|/)?)$',
    re.IGNORECASE,
)


def _clean_prose_segment(segment, strip_leading_ws=True):
    """Strip leaked citation tags, metadata lines, and orphaned spacing in prose."""
    text = _XML_WRAPPER_RE.sub('', segment)
    text = _META_LINE_RE.sub('', text)
    text = _CITATION_TAG_RE.sub('', text)
    text = _RAW_ID_SCORE_RE.sub('', text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r'\1', text)
    text = _MULTI_SPACE_RE.sub(' ', text)
    if strip_leading_ws:
        text = re.sub(r'\A[ \t]+', '', text)
    text = re.sub(r'(?<=\n)[ \t]+', '', text)
    return text


def sanitize_response(text):
    """Remove leaked technical artifacts while preserving fenced/inline code blocks."""
    if not text:
        return ''
    cleaned = _SPECIAL_TOKEN_RE.sub('', text)
    parts = _CODE_SEGMENT_RE.split(cleaned)
    out = [
        part if idx % 2 == 1 else _clean_prose_segment(part, strip_leading_ws=(idx == 0))
        for idx, part in enumerate(parts)
    ]
    joined = ''.join(out)
    joined = _MULTI_NEWLINE_RE.sub('\n\n', joined)
    return joined.strip()


class StreamingSanitizer:
    """Incrementally sanitize SSE delta chunks without leaking split tags/tokens."""

    def __init__(self):
        self._buf = ''
        self._in_fence = False

    def feed(self, chunk):
        """Process next text delta and return safe text ready to stream."""
        if not chunk:
            return ''
        data = self._buf + chunk
        self._buf = ''
        if self._in_fence:
            return self._process_in_fence(data)

        fence_idx = data.find('```')
        if fence_idx != -1:
            before = data[:fence_idx]
            rest = data[fence_idx:]
            emitted_before = self._clean_emit(before, is_final=True)
            self._in_fence = True
            return emitted_before + self._process_in_fence(rest)

        # Hold back potential trailing ``` start or partial artifact tag/token.
        hold_len = self._find_hold_length(data)
        if hold_len:
            self._buf = data[-hold_len:]
            data = data[:-hold_len]
        return self._clean_emit(data, is_final=False)

    def flush(self):
        """Flush any remaining buffered text at stream completion."""
        if not self._buf:
            return ''
        rem = self._buf
        self._buf = ''
        if self._in_fence:
            return _SPECIAL_TOKEN_RE.sub('', rem)
        return self._clean_emit(rem, is_final=True)

    def _process_in_fence(self, data):
        # Emit fenced code verbatim except stripped special tokens, tracking closing fence.
        close_search_start = 3 if data.startswith('```') and len(data) >= 3 else 0
        close_idx = data.find('```', close_search_start)
        if close_idx == -1:
            return _SPECIAL_TOKEN_RE.sub('', data)
        fence_end = close_idx + 3
        self._in_fence = False
        code_part = _SPECIAL_TOKEN_RE.sub('', data[:fence_end])
        after = data[fence_end:]
        return code_part + self.feed(after)

    @staticmethod
    def _find_hold_length(text):
        for backticks in ('``', '`'):
            if text.endswith(backticks):
                return len(backticks)
        match = _PARTIAL_TAIL_RE.search(text)
        if match:
            return len(match.group(0))
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith('\n'):
            last_line = lines[-1].lstrip()
            lowered = last_line.lower()
            for prefix in ('topik:', 'konteks sumber:', 'memori personal:', 'sumber referensi:'):
                if prefix.startswith(lowered) or (lowered.startswith(prefix) and len(last_line) < 160):
                    return len(lines[-1])
        return 0

    @staticmethod
    def _clean_emit(text, is_final=False):
        if not text:
            return ''
        cleaned = _SPECIAL_TOKEN_RE.sub('', text)
        parts = re.split(r'(`[^`\n]+`)', cleaned)
        out = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                out.append(part)
            else:
                seg = _XML_WRAPPER_RE.sub('', part)
                seg = _META_LINE_RE.sub('', seg)
                seg = _CITATION_TAG_RE.sub('', seg)
                seg = _RAW_ID_SCORE_RE.sub('', seg)
                seg = _SPACE_BEFORE_PUNCT_RE.sub(r'\1', seg)
                seg = _MULTI_SPACE_RE.sub(' ', seg)
                out.append(seg)
        result = ''.join(out)
        if is_final:
            result = _MULTI_NEWLINE_RE.sub('\n\n', result)
        return result
