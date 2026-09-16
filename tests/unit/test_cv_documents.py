"""Tests for in-memory CV document parsing."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from jobmarket.cv.documents import (
    DOCX_MIME_TYPE,
    PDF_MIME_TYPE,
    TXT_MIME_TYPE,
    EmptyDocumentError,
    ExcessivePageCountError,
    MalformedDocumentError,
    OversizedDocumentError,
    UnsupportedDocumentError,
    parse_document,
)


def test_txt_parsing_normalizes_line_endings_and_hashes(tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_bytes(b"Python\r\nAWS\rData Science")

    document = parse_document(path)

    assert document.filename == "cv.txt"
    assert document.mime_type == TXT_MIME_TYPE
    assert document.text == "Python\nAWS\nData Science"
    assert document.page_count is None
    assert len(document.file_hash) == 64
    assert document.pages[0].page is None


def test_docx_parsing_with_zip_xml_fallback(tmp_path: Path) -> None:
    path = tmp_path / "cv.docx"
    _write_docx(path, ["Data Scientist", "Python and AWS"])

    document = parse_document(path)

    assert document.mime_type == DOCX_MIME_TYPE
    assert "Data Scientist" in document.text
    assert "Python and AWS" in document.text


def test_pdf_parsing_preserves_page_offsets(tmp_path: Path) -> None:
    path = tmp_path / "cv.pdf"
    path.write_bytes(_minimal_pdf(["Python AWS", "Data Science"]))

    document = parse_document(path)

    assert document.mime_type == PDF_MIME_TYPE
    assert document.page_count == 2
    assert "Python AWS" in document.text
    assert "Data Science" in document.text
    assert document.pages[0].page == 1
    assert document.pages[1].page == 2
    assert document.text[document.pages[1].start_offset : document.pages[1].end_offset].startswith(
        "Data Science"
    )


def test_empty_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        parse_document(path)


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cv.rtf"
    path.write_text("Python", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError):
        parse_document(path)


def test_malformed_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"%PDF not really a pdf")

    with pytest.raises(MalformedDocumentError):
        parse_document(path)


def test_malformed_docx_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.docx"
    path.write_bytes(b"not a zip")

    with pytest.raises(MalformedDocumentError):
        parse_document(path)


def test_file_size_limit_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("Python", encoding="utf-8")

    with pytest.raises(OversizedDocumentError):
        parse_document(path, max_file_size_bytes=2)


def test_pdf_page_limit_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "too-many.pdf"
    path.write_bytes(_minimal_pdf(["one", "two"]))

    with pytest.raises(ExcessivePageCountError):
        parse_document(path, max_pdf_pages=1)


_DOCX_CONTENT_TYPES = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>"
    "<Default Extension='rels' "
    "ContentType='application/vnd.openxmlformats-package.relationships+xml'/>"
    "<Default Extension='xml' ContentType='application/xml'/>"
    "<Override PartName='/word/document.xml' ContentType="
    "'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml'/>"
    "</Types>"
)
_DOCX_ROOT_RELS = (
    "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
    "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"
    "<Relationship Id='rId1' Type="
    "'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' "
    "Target='word/document.xml'/>"
    "</Relationships>"
)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    """A minimal but spec-valid OPC package — python-docx also requires
    `[Content_Types].xml` and `_rels/.rels`, not just `word/document.xml`."""
    body = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs
    )
    xml = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        "<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'>"
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", _DOCX_CONTENT_TYPES)
        archive.writestr("_rels/.rels", _DOCX_ROOT_RELS)
        archive.writestr("word/document.xml", xml)


def _minimal_pdf(page_texts: list[str]) -> bytes:
    objects = ["1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj"]
    kids = " ".join(f"{3 + index * 2} 0 R" for index in range(len(page_texts)))
    objects.append(f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >> endobj")
    for index, text in enumerate(page_texts):
        page_id = 3 + index * 2
        content_id = page_id + 1
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects.append(
            f"{page_id} 0 obj << /Type /Page /Parent 2 0 R /Resources "
            f"<< /Font << /F1 99 0 R >> >> /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R >> endobj"
        )
        objects.append(
            f"{content_id} 0 obj << /Length {len(stream.encode())} >> stream\n"
            f"{stream}\nendstream endobj"
        )
    objects.append("99 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    content = "%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(content.encode()))
        content += obj + "\n"
    xref_offset = len(content.encode())
    max_obj = 99
    entries = ["0000000000 65535 f "] + ["0000000000 00000 f "] * max_obj
    for obj, offset in zip(objects, offsets[1:], strict=True):
        obj_id = int(obj.split()[0])
        entries[obj_id] = f"{offset:010d} 00000 n "
    content += f"xref\n0 {max_obj + 1}\n" + "\n".join(entries) + "\n"
    content += f"trailer << /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    return content.encode()
