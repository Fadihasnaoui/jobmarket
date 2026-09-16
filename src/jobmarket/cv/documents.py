"""Safe in-memory CV document parsing for TXT, PDF, and DOCX."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from jobmarket.cv.profile import (
    PARSER_VERSION,
    DocumentPage,
    DocumentTextBlock,
    ExtractionWarning,
    ParsedDocument,
)

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 20
PDF_MIME_TYPE = "application/pdf"
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
TXT_MIME_TYPE = "text/plain"

_PAGE_SEPARATOR = "\n\n"
_WORD_TEXT_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
_WORD_PARAGRAPH_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"


class CvDocumentError(ValueError):
    """Base class for CV document parsing failures."""


class UnsupportedDocumentError(CvDocumentError):
    """Raised when the file extension or format is unsupported."""


class OversizedDocumentError(CvDocumentError):
    """Raised when a document exceeds the configured size limit."""


class MalformedDocumentError(CvDocumentError):
    """Raised when a supported document cannot be parsed safely."""


class EmptyDocumentError(CvDocumentError):
    """Raised when parsing succeeds but no text is extracted."""


class ExcessivePageCountError(CvDocumentError):
    """Raised when a PDF exceeds the configured page-count limit."""


def parse_document(
    path: str | Path,
    *,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
    max_pdf_pages: int = MAX_PDF_PAGES,
) -> ParsedDocument:
    """Parse a CV document into normalized text and deterministic metadata."""
    document_path = Path(path)
    if not document_path.is_file():
        raise UnsupportedDocumentError(f"Document does not exist or is not a file: {document_path}")

    raw = document_path.read_bytes()
    file_size = len(raw)
    if file_size > max_file_size_bytes:
        raise OversizedDocumentError(
            f"Document size {file_size} bytes exceeds limit {max_file_size_bytes} bytes"
        )

    extension = document_path.suffix.casefold()
    file_hash = hashlib.sha256(raw).hexdigest()
    if extension == ".txt":
        text = _parse_txt(raw)
        return _build_document(
            filename=document_path.name,
            mime_type=TXT_MIME_TYPE,
            text=text,
            page_count=None,
            page_texts=[(None, text)],
            file_hash=file_hash,
            file_size=file_size,
            warnings=[],
            blocks=[],
        )
    if extension == ".pdf":
        if not raw.startswith(b"%PDF"):
            raise MalformedDocumentError("PDF file does not start with a PDF header")
        page_texts, warnings, blocks = _parse_pdf(raw, max_pdf_pages=max_pdf_pages)
        return _build_document(
            filename=document_path.name,
            mime_type=PDF_MIME_TYPE,
            text=None,
            page_count=len(page_texts),
            page_texts=[(idx + 1, page_text) for idx, page_text in enumerate(page_texts)],
            file_hash=file_hash,
            file_size=file_size,
            warnings=warnings,
            blocks=blocks,
        )
    if extension == ".docx":
        if not zipfile.is_zipfile(io.BytesIO(raw)):
            raise MalformedDocumentError("DOCX file is not a valid ZIP container")
        text = _parse_docx(raw)
        return _build_document(
            filename=document_path.name,
            mime_type=DOCX_MIME_TYPE,
            text=text,
            page_count=None,
            page_texts=[(None, text)],
            file_hash=file_hash,
            file_size=file_size,
            warnings=[],
            blocks=[],
        )
    raise UnsupportedDocumentError(f"Unsupported CV document extension: {extension or '(none)'}")


def _parse_txt(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MalformedDocumentError("TXT document must be valid UTF-8") from exc


def _parse_pdf(
    raw: bytes, *, max_pdf_pages: int
) -> tuple[list[str], list[ExtractionWarning], list[DocumentTextBlock]]:
    try:
        reader = PdfReader(io.BytesIO(raw), strict=False)
        page_count = len(reader.pages)
    except (PdfReadError, OSError, ValueError, KeyError) as exc:
        raise MalformedDocumentError("Malformed PDF document") from exc
    if page_count > max_pdf_pages:
        raise ExcessivePageCountError(f"PDF page count {page_count} exceeds limit {max_pdf_pages}")

    warnings: list[ExtractionWarning] = []
    page_texts: list[str] = []
    all_blocks: list[DocumentTextBlock] = []
    for index, page in enumerate(reader.pages, start=1):
        blocks = _extract_positioned_pdf_blocks(page, page_number=index)
        if blocks:
            page_text = _reconstruct_page_reading_order(blocks)
            all_blocks.extend(blocks)
            warnings.append(
                ExtractionWarning(
                    code="pdf_layout_extraction",
                    message=f"PDF page {index} used positioned text reconstruction",
                )
            )
        else:
            try:
                page_text = page.extract_text() or ""
            except (PdfReadError, OSError, ValueError, KeyError) as exc:
                raise MalformedDocumentError(
                    f"Could not extract text from PDF page {index}"
                ) from exc
        normalized = _normalize_line_endings(page_text).strip()
        if not normalized:
            warnings.append(
                ExtractionWarning(code="empty_pdf_page", message=f"PDF page {index} has no text")
            )
        page_texts.append(normalized)
    return page_texts, warnings, all_blocks


def _extract_positioned_pdf_blocks(page: Any, *, page_number: int) -> list[DocumentTextBlock]:
    blocks: list[DocumentTextBlock] = []
    last_position = (0.0, 0.0)

    def visitor_text(
        text: str,
        _cm: object,
        tm: Any,
        _font_dict: object,
        font_size: Any,
    ) -> None:
        nonlocal last_position
        cleaned = _normalize_pdf_text(text)
        if not cleaned:
            return
        try:
            matrix = list(tm)  # pypdf passes a six-value text matrix.
            x = float(matrix[4])
            y = float(matrix[5])
            if (x, y) == (0.0, 0.0) and blocks:
                # Confirmed on a real Chromium-rendered CV: for certain spans
                # (right-aligned/flex-positioned text, e.g. a date range flush
                # right on a job's title row) pypdf hands back an identity text
                # matrix carrying no real translation, instead of the element's
                # actual position -- every other text run in the same document
                # gets a real (x, y). Falling back to a literal (0, 0) here
                # would inject a false "top-left of the page" signal that
                # corrupts row/region detection for the entire document (this
                # is exactly what produced the "PREVIOUS EXPERIENCE"/date-range
                # scrambling documented in docs/LIMITATIONS.md). Inheriting the
                # previous successfully-positioned block's coordinates instead
                # keeps it on the same row as whatever it was rendered
                # immediately after in the content stream -- reliably its true
                # row partner for this class of PDF.
                x, y = last_position
            else:
                last_position = (x, y)
        except (TypeError, ValueError, IndexError):
            x, y = last_position
        try:
            size = max(1.0, float(font_size))
        except (TypeError, ValueError):
            size = 10.0
        block_id = f"p{page_number}-b{len(blocks) + 1:04d}"
        blocks.append(
            DocumentTextBlock(
                page=page_number,
                block_id=block_id,
                x=round(x, 2),
                y=round(y, 2),
                width=round(max(size, len(cleaned) * size * 0.48), 2),
                height=round(size, 2),
                text=cleaned,
            )
        )

    try:
        page.extract_text(visitor_text=visitor_text)
    except (PdfReadError, OSError, ValueError, KeyError, TypeError):
        return []
    return [block for block in blocks if block.text.strip()]


_ROW_Y_TOLERANCE = 6.0  # pt; blocks within this y distance are the same visual line


def _reconstruct_page_reading_order(blocks: list[DocumentTextBlock]) -> str:
    ordered = order_layout_blocks(blocks)
    y_key = _y_key(blocks)
    lines: list[str] = []
    prev: DocumentTextBlock | None = None
    for block in ordered:
        text = block.text.strip()
        if not text:
            continue
        if prev is not None and abs(y_key(block) - y_key(prev)) <= _ROW_Y_TOLERANCE:
            # Same visual line as the previous block (e.g. a title and a
            # right-flush date range) -- keep them on one reconstructed line
            # instead of scattering them across separate lines, so a downstream
            # extractor (deterministic or LLM) can actually see them together.
            lines[-1] = f"{lines[-1]} {text}"
        else:
            lines.append(text)
        prev = block
    return "\n".join(lines)


def _y_increases_downward(blocks: list[DocumentTextBlock]) -> bool:
    """Detect whether this PDF's y-coordinates increase toward the bottom of the
    page (seen from real-world PDFs generated by headless-browser/Chromium-style
    print-to-PDF pipelines -- confirmed by direct inspection: a Chromium-rendered
    CV had its page title at y=40 and its page bottom around y=560) or decrease
    toward the bottom (the traditional PDF convention: bottom-left origin, y-up,
    which the rest of this module and its tests assume).

    Every real PDF generator, regardless of which axis convention it uses, still
    emits text roughly in natural top-to-bottom reading order within the content
    stream -- so comparing the mean y of the first third of blocks (in original
    extraction order) against the last third tells us which direction is "down"
    for this specific document. Getting this wrong previously produced completely
    reversed/scrambled reading order for an entire class of real CVs (browser-
    rendered resume-builder templates), which the downstream LLM extraction had
    no way to recover from -- see docs/LIMITATIONS.md.

    Uses a third of the blocks rather than just the first/last one, since a
    single endpoint is often a sidebar or header element rendered out of the
    dominant top-to-bottom flow.
    """
    if len(blocks) < 6:
        return False
    third = max(1, len(blocks) // 3)
    first_mean = sum(b.y for b in blocks[:third]) / third
    last_mean = sum(b.y for b in blocks[-third:]) / third
    return last_mean > first_mean


def _y_key(blocks: list[DocumentTextBlock]) -> Callable[[DocumentTextBlock], float]:
    downward = _y_increases_downward(blocks)
    return (lambda b: b.y) if downward else (lambda b: -b.y)


_REGION_MIN_GAP = 90.0  # pt; a genuine column gutter (sidebar vs. main), not same-row spacing


def _cluster_regions(blocks: list[DocumentTextBlock]) -> list[list[DocumentTextBlock]]:
    """Split blocks into a small number of major regions (e.g. main column vs.
    sidebar) using the largest gaps in their x-distribution, rather than a fixed
    per-block tolerance.

    A fixed tolerance can't tell apart two genuinely distinct regions (typically
    150-300+pt apart, e.g. a sidebar gutter) from same-row text that's merely
    right-flush (routinely 100-400+pt from its row partner -- a title on the left
    and a date range flush right on the same line). Using a big-gap threshold
    instead means same-row content stays in one region and gets reassembled by
    `_rows_in_reading_order` below; only genuine column boundaries get cut.
    """
    if not blocks:
        return []
    distinct_x = sorted({round(b.x) for b in blocks})
    cut_points = [
        (a + b) / 2
        for a, b in zip(distinct_x, distinct_x[1:], strict=False)
        if b - a >= _REGION_MIN_GAP
    ]
    regions: list[list[DocumentTextBlock]] = [[] for _ in range(len(cut_points) + 1)]
    for block in blocks:
        idx = sum(1 for cut in cut_points if block.x >= cut)
        regions[idx].append(block)
    return [region for region in regions if region]


def _reattach_minor_regions(
    regions: list[list[DocumentTextBlock]], y_key: Callable[[DocumentTextBlock], float]
) -> list[list[DocumentTextBlock]]:
    """A region whose EVERY block aligns row-for-row with blocks in other regions
    is almost certainly same-row metadata (a right-flush date range, or a
    right-flush company on a compact "Title ... Company Date" row) rather than a
    genuine structural column of its own -- a real prose column (however short)
    essentially never lands every one of its lines on exactly the same rows as
    another column's content, while a metadata column that exists specifically to
    annotate each row of another column does, by construction. This is a
    row-alignment test, not a size threshold, since a real date column can easily
    have as many or more blocks than a short genuine column.

    Reassigns each such region's blocks into whichever other region shares that
    row, so it gets read together with its row instead of emitted as its own
    isolated, order-scrambled fragment -- see docs/LIMITATIONS.md.
    """
    if len(regions) <= 1:
        return regions

    absorbable = {
        i
        for i, region in enumerate(regions)
        if all(
            _closest_row_major(
                block, {j: regions[j] for j in range(len(regions)) if j != i}, y_key
            )
            is not None
            for block in region
        )
    }
    if not absorbable or absorbable == set(range(len(regions))):
        return regions

    kept = {i: list(regions[i]) for i in range(len(regions)) if i not in absorbable}
    leftover: list[DocumentTextBlock] = []
    for i in absorbable:
        for block in regions[i]:
            target = _closest_row_major(block, kept, y_key)
            if target is not None:
                kept[target].append(block)
            else:
                leftover.append(block)

    result = [kept[i] for i in sorted(kept)]
    if leftover:
        result.append(leftover)
    return result


def _closest_row_major(
    block: DocumentTextBlock,
    majors: dict[int, list[DocumentTextBlock]],
    y_key: Callable[[DocumentTextBlock], float],
) -> int | None:
    best_idx: int | None = None
    best_dy = _ROW_Y_TOLERANCE
    for idx, region in majors.items():
        for other in region:
            dy = abs(y_key(block) - y_key(other))
            if dy <= best_dy:
                best_dy = dy
                best_idx = idx
    return best_idx


def _rows_in_reading_order(
    region: list[DocumentTextBlock], y_key: Callable[[DocumentTextBlock], float]
) -> list[DocumentTextBlock]:
    """Within one region, group blocks into visual rows (y within
    `_ROW_Y_TOLERANCE`) and read each row left-to-right -- so a title, an inline
    company, and a right-flush date range on the same line come out adjacent in
    the reconstructed text instead of being torn apart by column-level x-jumps.
    """
    sorted_by_y = sorted(region, key=lambda b: (y_key(b), b.x, b.block_id))
    rows: list[list[DocumentTextBlock]] = []
    for block in sorted_by_y:
        if rows and abs(y_key(block) - y_key(rows[-1][0])) <= _ROW_Y_TOLERANCE:
            rows[-1].append(block)
        else:
            rows.append([block])
    return [block for row in rows for block in sorted(row, key=lambda b: (b.x, b.block_id))]


def order_layout_blocks(blocks: list[DocumentTextBlock]) -> list[DocumentTextBlock]:
    """Return deterministic, row-aware reading order for positioned PDF blocks."""
    if not blocks:
        return []
    y_key = _y_key(blocks)
    regions = _cluster_regions(blocks)
    regions = _reattach_minor_regions(regions, y_key)
    regions.sort(key=lambda region: min(block.x for block in region))
    return [block for region in regions for block in _rows_in_reading_order(region, y_key)]


def _normalize_pdf_text(text: str) -> str:
    cleaned = _normalize_line_endings(text)
    cleaned = cleaned.replace("\xa0", " ")
    cleaned = cleaned.replace("\ufb01", "fi").replace("\ufb02", "fl")
    cleaned = " ".join(part.strip() for part in cleaned.splitlines() if part.strip())
    return _collapse_spaced_glyphs(cleaned).strip()


def _collapse_spaced_glyphs(text: str) -> str:
    parts = re.split(r"(\s{2,})", text.strip())
    collapsed: list[str] = []
    for part in parts:
        if not part.strip():
            collapsed.append(" ")
            continue
        tokens = part.split()
        if len(tokens) >= 3 and sum(len(token) == 1 for token in tokens) / len(tokens) >= 0.8:
            collapsed.append("".join(tokens))
        else:
            collapsed.append(part.strip())
    return re.sub(r"\s+", " ", "".join(collapsed)).strip()


def _parse_docx(raw: bytes) -> str:
    text = _parse_docx_with_python_docx(raw)
    if text is not None:
        return text
    return _parse_docx_zip_xml(raw)


def _parse_docx_with_python_docx(raw: bytes) -> str | None:
    if importlib.util.find_spec("docx") is None:
        return None
    try:
        docx_module = importlib.import_module("docx")
        document = docx_module.Document(io.BytesIO(raw))
    except Exception as exc:  # pragma: no cover - exact library exceptions vary by version.
        raise MalformedDocumentError("Malformed DOCX document") from exc
    paragraphs = [str(paragraph.text) for paragraph in document.paragraphs]
    return "\n".join(paragraphs)


def _parse_docx_zip_xml(raw: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            document_xml = archive.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise MalformedDocumentError("DOCX file is missing word/document.xml") from exc
    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise MalformedDocumentError("DOCX document.xml is malformed") from exc

    paragraphs: list[str] = []
    for paragraph in root.iter(_WORD_PARAGRAPH_NS):
        text = "".join(node.text or "" for node in paragraph.iter(_WORD_TEXT_NS))
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _build_document(
    *,
    filename: str,
    mime_type: str,
    text: str | None,
    page_count: int | None,
    page_texts: list[tuple[int | None, str]],
    file_hash: str,
    file_size: int,
    warnings: list[ExtractionWarning],
    blocks: list[DocumentTextBlock],
) -> ParsedDocument:
    pages: list[DocumentPage] = []
    parts: list[str] = []
    offset = 0
    for index, (page_number, page_text) in enumerate(page_texts):
        if index:
            parts.append(_PAGE_SEPARATOR)
            offset += len(_PAGE_SEPARATOR)
        normalized_page = _normalize_line_endings(page_text).strip()
        start = offset
        parts.append(normalized_page)
        offset += len(normalized_page)
        pages.append(
            DocumentPage(
                page=page_number,
                text=normalized_page,
                start_offset=start,
                end_offset=offset,
            )
        )
    combined_text = _normalize_line_endings(text if text is not None else "".join(parts)).strip()
    if text is None:
        combined_text = "".join(parts).strip()
    if not combined_text:
        raise EmptyDocumentError("Document contains no extractable text")
    if len(pages) == 1 and pages[0].page is None:
        pages = [
            pages[0].model_copy(
                update={
                    "text": combined_text,
                    "start_offset": 0,
                    "end_offset": len(combined_text),
                }
            )
        ]
    adjusted_blocks = _assign_block_offsets(blocks, combined_text)
    return ParsedDocument(
        filename=filename,
        mime_type=mime_type,
        parser_version=PARSER_VERSION,
        text=combined_text,
        page_count=page_count,
        pages=pages,
        blocks=adjusted_blocks,
        warnings=warnings,
        file_hash=file_hash,
        file_size=file_size,
    )


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _assign_block_offsets(
    blocks: list[DocumentTextBlock], combined_text: str
) -> list[DocumentTextBlock]:
    adjusted: list[DocumentTextBlock] = []
    cursor = 0
    for block in order_layout_blocks(blocks):
        text = block.text.strip()
        start = combined_text.find(text, cursor)
        if start < 0:
            start = combined_text.find(text)
        if start < 0:
            adjusted.append(block)
            continue
        end = start + len(text)
        adjusted.append(block.model_copy(update={"start_offset": start, "end_offset": end}))
        cursor = end
    return adjusted
