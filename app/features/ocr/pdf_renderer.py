"""Document page renderer and structured extractor (PDF, DOCX, HTML, and images)."""

import html
from html.parser import HTMLParser
import os
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

try:
    import pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

try:
    import pymupdf4llm
except ImportError:  # pragma: no cover
    pymupdf4llm = None

from app.features.ocr.exceptions import DocumentValidationError, OcrUnavailableError

SUPPORTED_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
SUPPORTED_DOCUMENT_SUFFIXES = {'.pdf', '.docx', '.html', '.htm'}
SUPPORTED_SUFFIXES = {*SUPPORTED_DOCUMENT_SUFFIXES, *SUPPORTED_IMAGE_SUFFIXES}
DEFAULT_MAX_FILE_MB = 20
DEFAULT_MAX_PAGES = 30
DEFAULT_RENDER_DPI = 150
DEFAULT_TEXT_PAGE_CHARS = 3500
_CID_GARBLE_RE = re.compile(r'\(cid:\d+\)')


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    image_path: Path | None
    native_text: str = ''
    is_structured_text: bool = False


def max_upload_bytes():
    mb = int(os.getenv('OCR_MAX_FILE_MB', str(DEFAULT_MAX_FILE_MB)))
    return max(1, mb) * 1024 * 1024


def max_document_pages():
    return max(1, int(os.getenv('OCR_MAX_PAGES', str(DEFAULT_MAX_PAGES))))


def validate_upload_metadata(filename, size_bytes, max_bytes=None):
    clean_name = Path(filename or '').name.strip()
    if not clean_name:
        raise DocumentValidationError('Nama file dokumen tidak valid.')
    suffix = Path(clean_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ', '.join(sorted(SUPPORTED_SUFFIXES))
        raise DocumentValidationError(
            f'Format file {suffix or "(tanpa ekstensi)"} tidak didukung. Gunakan: {allowed}.'
        )
    if size_bytes <= 0:
        raise DocumentValidationError('File dokumen kosong.')
    limit = max_bytes if max_bytes is not None else max_upload_bytes()
    if size_bytes > limit:
        limit_mb = round(limit / (1024 * 1024), 1)
        raise DocumentValidationError(
            f'Ukuran file melebihi batas maksimum {limit_mb} MB.'
        )
    return clean_name, suffix


def _paginate_markdown_blocks(blocks, *, max_chars=DEFAULT_TEXT_PAGE_CHARS, page_limit=DEFAULT_MAX_PAGES):
    """Group logical Markdown blocks into bounded virtual pages."""
    pages = []
    current = []
    cur_len = 0

    def _flush():
        nonlocal current, cur_len
        text = '\n\n'.join(b for b in current if b.strip()).strip()
        if text:
            pages.append(text)
        current = []
        cur_len = 0

    for block in blocks:
        if block == '\f':
            _flush()
            continue
        cleaned = (block or '').strip()
        if not cleaned:
            continue
        if current and cur_len + len(cleaned) + 2 > max_chars:
            _flush()
        current.append(cleaned)
        cur_len += len(cleaned) + 2

    _flush()
    if len(pages) > page_limit:
        raise DocumentValidationError(
            f'Jumlah bagian/halaman dokumen ({len(pages)}) melebihi batas maksimum {page_limit}.'
        )
    return [
        RenderedPage(page_number=idx, image_path=None, native_text=md, is_structured_text=True)
        for idx, md in enumerate(pages, start=1)
    ]


def _local_tag(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def _format_markdown_table(rows):
    """Format a 2D list of strings as a clean GitHub-flavored Markdown table."""
    clean_rows = [[re.sub(r'\s+', ' ', cell).replace('|', '\\|').strip() for cell in row] for row in rows]
    clean_rows = [row for row in clean_rows if any(row)]
    if not clean_rows:
        return ''
    col_count = max(len(r) for r in clean_rows)
    padded = [r + [''] * (col_count - len(r)) for r in clean_rows]
    lines = ['| ' + ' | '.join(padded[0]) + ' |', '| ' + ' | '.join(['---'] * col_count) + ' |']
    for row in padded[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def _extract_docx_paragraph(p_elem):
    """Extract paragraph text, heading level, list prefix, and page-break flag from w:p."""
    texts = []
    has_page_break = False
    for node in p_elem.iter():
        tag = _local_tag(node.tag)
        if tag == 't' and node.text:
            texts.append(node.text)
        elif tag == 'tab':
            texts.append(' ')
        elif tag == 'br':
            br_type = next((v for k, v in node.attrib.items() if _local_tag(k) == 'type'), '')
            if br_type == 'page':
                has_page_break = True
            else:
                texts.append('\n')

    raw_text = re.sub(r'[ \t]+', ' ', ''.join(texts)).strip()
    if not raw_text:
        return '', has_page_break

    p_style = ''
    is_list = False
    p_pr = next((c for c in p_elem if _local_tag(c.tag) == 'pPr'), None)
    if p_pr is not None:
        for child in p_pr:
            ctag = _local_tag(child.tag)
            if ctag == 'pStyle':
                p_style = next((v for k, v in child.attrib.items() if _local_tag(k) == 'val'), '')
            elif ctag == 'numPr':
                is_list = True

    style_lower = p_style.lower()
    heading_level = 0
    if style_lower == 'title':
        heading_level = 1
    elif style_lower == 'subtitle':
        heading_level = 2
    else:
        m = re.search(r'(?:heading|judul)\s*([1-4])', style_lower)
        if m:
            heading_level = int(m.group(1))

    if heading_level:
        return f'{"#" * heading_level} {raw_text}', has_page_break
    if is_list and not raw_text.startswith(('-', '*', '•')):
        return f'- {raw_text}', has_page_break
    return raw_text, has_page_break


def _extract_docx_pages(source: Path, *, page_limit: int):
    try:
        with zipfile.ZipFile(source, 'r') as zf:
            if 'word/document.xml' not in zf.namelist():
                raise DocumentValidationError('File DOCX tidak valid (word/document.xml tidak ditemukan).')
            xml_bytes = zf.read('word/document.xml')
    except DocumentValidationError:
        raise
    except Exception as exc:
        raise DocumentValidationError(f'File DOCX rusak atau tidak dapat dibuka: {exc}') from exc

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise DocumentValidationError(f'XML di dalam DOCX rusak: {exc}') from exc

    body = next((el for el in root.iter() if _local_tag(el.tag) == 'body'), None)
    if body is None:
        raise DocumentValidationError('Isi dokumen DOCX kosong.')

    blocks = []
    for child in body:
        tag = _local_tag(child.tag)
        if tag == 'p':
            text, has_break = _extract_docx_paragraph(child)
            if text:
                blocks.append(text)
            if has_break:
                blocks.append('\f')
        elif tag == 'tbl':
            rows = []
            for tr in child:
                if _local_tag(tr.tag) != 'tr':
                    continue
                row_cells = []
                for tc in tr:
                    if _local_tag(tc.tag) != 'tc':
                        continue
                    cell_parts = []
                    for p in tc:
                        if _local_tag(p.tag) == 'p':
                            pt, _ = _extract_docx_paragraph(p)
                            if pt:
                                cell_parts.append(pt.lstrip('#').strip())
                    row_cells.append(' '.join(cell_parts))
                if row_cells:
                    rows.append(row_cells)
            table_md = _format_markdown_table(rows)
            if table_md:
                blocks.append(table_md)

    pages = _paginate_markdown_blocks(blocks, page_limit=page_limit)
    if not pages:
        raise DocumentValidationError('Dokumen DOCX tidak memiliki teks yang dapat dibaca.')
    return pages


class _HtmlToMarkdownParser(HTMLParser):
    _SKIP_TAGS = {'script', 'style', 'noscript', 'svg', 'template', 'iframe'}
    _BLOCK_TAGS = {'p', 'div', 'section', 'article', 'main', 'blockquote', 'li', 'h1', 'h2', 'h3', 'h4', 'pre'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.in_pre = False
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.list_depth = 0
        self.heading_level = 0
        self.cur_buf = []
        self.cur_row = []
        self.cur_table = []
        self.blocks = []
        self.title = ''
        self.in_title = False

    def _flush_buf(self):
        raw = ''.join(self.cur_buf)
        self.cur_buf = []
        if self.in_pre:
            cleaned = raw.strip('\n')
            if cleaned.strip():
                self.blocks.append(f'```\n{cleaned}\n```')
            return
        text = re.sub(r'[ \t]+', ' ', raw)
        text = re.sub(r'\n{2,}', '\n', text).strip()
        if not text:
            return
        if self.heading_level:
            self.blocks.append(f'{"#" * self.heading_level} {text}')
        else:
            self.blocks.append(text)

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if t == 'title':
            self.in_title = True
            self.cur_buf = []
            return
        if t in ('ul', 'ol'):
            self._flush_buf()
            self.list_depth += 1
            return
        if t == 'table':
            self._flush_buf()
            self.in_table = True
            self.cur_table = []
            return
        if t == 'tr' and self.in_table:
            self.in_row = True
            self.cur_row = []
            return
        if t in ('th', 'td') and self.in_row:
            self.in_cell = True
            self.cur_buf = []
            return
        if t == 'br':
            self.cur_buf.append('\n')
            return
        if t in self._BLOCK_TAGS:
            self._flush_buf()
            if t in ('h1', 'h2', 'h3', 'h4'):
                self.heading_level = int(t[1])
            elif t == 'li':
                indent = '  ' * max(0, self.list_depth - 1)
                self.cur_buf.append(f'{indent}- ')
            elif t == 'pre':
                self.in_pre = True

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self._SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if t == 'title':
            self.in_title = False
            self.title = re.sub(r'\s+', ' ', ''.join(self.cur_buf)).strip()
            self.cur_buf = []
            return
        if t in ('ul', 'ol'):
            self._flush_buf()
            self.list_depth = max(0, self.list_depth - 1)
            return
        if t in ('th', 'td') and self.in_cell:
            cell = re.sub(r'\s+', ' ', ''.join(self.cur_buf)).strip()
            self.cur_row.append(cell)
            self.cur_buf = []
            self.in_cell = False
            return
        if t == 'tr' and self.in_row:
            if any(self.cur_row):
                self.cur_table.append(self.cur_row)
            self.cur_row = []
            self.in_row = False
            return
        if t == 'table' and self.in_table:
            self.in_table = False
            md = _format_markdown_table(self.cur_table)
            self.cur_table = []
            if md:
                self.blocks.append(md)
            return
        if t in self._BLOCK_TAGS:
            self._flush_buf()
            if t in ('h1', 'h2', 'h3', 'h4'):
                self.heading_level = 0
            elif t == 'pre':
                self.in_pre = False

    def handle_data(self, data):
        if self.skip_depth:
            return
        self.cur_buf.append(html.unescape(data))

    def finish(self):
        self._flush_buf()
        if self.title and (not self.blocks or not self.blocks[0].startswith('#')):
            return [f'# {self.title}', *self.blocks]
        return self.blocks


def _extract_html_pages(source: Path, *, page_limit: int):
    raw_bytes = source.read_bytes()
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1'):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw_bytes.decode('utf-8', errors='replace')

    parser = _HtmlToMarkdownParser()
    parser.feed(text)
    parser.close()
    blocks = parser.finish()
    pages = _paginate_markdown_blocks(blocks, page_limit=page_limit)
    if not pages:
        raise DocumentValidationError('File HTML tidak memiliki teks yang dapat dibaca.')
    return pages


def _is_clean_digital_pdf_page(page, native_text: str) -> bool:
    """Return True when a PDF page contains reliable embedded digital text (not scanned/image-only)."""
    clean = re.sub(r'\s+', ' ', native_text or '').strip()
    if len(clean) < 120:
        return False
    if _CID_GARBLE_RE.search(clean):
        return False
    words = [w for w in clean.split() if any(ch.isalnum() for ch in w)]
    if len(words) < 15:
        return False
    alnum = sum(1 for ch in clean if ch.isalnum())
    if alnum / max(1, len(clean)) < 0.45:
        return False
    has_images = bool(page.get_images())
    return (not has_images) or len(clean) >= 350


def _extract_pymupdf4llm_pages(source: Path, page_numbers: list[int]) -> dict[int, str]:
    if pymupdf4llm is None or not page_numbers:
        return {}
    try:
        zero_based = [p - 1 for p in page_numbers]
        chunks = pymupdf4llm.to_markdown(str(source), pages=zero_based, page_chunks=True)
        out = {}
        for idx, item in enumerate(chunks):
            pnum = page_numbers[idx] if idx < len(page_numbers) else None
            if pnum is not None and isinstance(item, dict):
                md = (item.get('text') or '').strip()
                if md:
                    out[pnum] = md
        return out
    except Exception:
        return {}


def render_document_pages(file_path, work_dir, *, dpi=None, max_pages=None):
    source = Path(file_path)
    if not source.exists() or source.stat().st_size <= 0:
        raise DocumentValidationError('File dokumen tidak ditemukan atau kosong.')

    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentValidationError(f'Format file {suffix} tidak didukung.')

    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_limit = max_pages if max_pages is not None else max_document_pages()

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        target = out_dir / f'page_0001{suffix}'
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return [RenderedPage(page_number=1, image_path=target, native_text='')]

    if suffix == '.docx':
        return _extract_docx_pages(source, page_limit=page_limit)

    if suffix in ('.html', '.htm'):
        return _extract_html_pages(source, page_limit=page_limit)

    if pymupdf is None:
        raise OcrUnavailableError(
            "Pustaka 'pymupdf' belum terpasang. Jalankan: pip install -r requirements-ocr.txt"
        )

    render_dpi = int(dpi or os.getenv('OCR_RENDER_DPI', str(DEFAULT_RENDER_DPI)))
    pdf_mode = os.getenv('OCR_PDF_MODE', 'hybrid').strip().lower()

    try:
        doc = pymupdf.open(str(source))
    except Exception as exc:
        raise DocumentValidationError(f'File PDF rusak atau tidak dapat dibuka: {exc}') from exc

    try:
        if doc.is_encrypted:
            raise DocumentValidationError('File PDF terkunci password dan tidak dapat diproses.')
        total = len(doc)
        if total < 1:
            raise DocumentValidationError('File PDF tidak memiliki halaman.')
        if total > page_limit:
            raise DocumentValidationError(
                f'Jumlah halaman PDF ({total}) melebihi batas maksimum {page_limit} halaman.'
            )

        digital_pages = []
        page_entries = []
        for idx, page in enumerate(doc, start=1):
            native_text = (page.get_text('text') or '').strip()
            use_digital = (
                pdf_mode != 'ocr_only'
                and _is_clean_digital_pdf_page(page, native_text)
            )
            if use_digital:
                digital_pages.append(idx)
                page_entries.append((idx, None, native_text, True))
            else:
                pix = page.get_pixmap(dpi=render_dpi)
                img_path = out_dir / f'page_{idx:04d}.png'
                pix.save(str(img_path))
                page_entries.append((idx, img_path, native_text, False))

        md_map = _extract_pymupdf4llm_pages(source, digital_pages) if digital_pages else {}
        rendered = []
        for idx, img_path, native_text, is_structured in page_entries:
            text_val = md_map.get(idx, native_text) if is_structured else native_text
            rendered.append(
                RenderedPage(
                    page_number=idx,
                    image_path=img_path,
                    native_text=text_val,
                    is_structured_text=is_structured,
                )
            )
        return rendered
    finally:
        doc.close()

