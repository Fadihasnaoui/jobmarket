"""Shared text and salary normalization for the parse stage."""

from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

import ftfy

_SENIORITY_PATTERN = re.compile(
    r"\b(senior|junior|sr\.?|jr\.?)\b|\((h/f|m/f/d|f/m/d|h/f/m)\)",
    re.IGNORECASE,
)

_LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b("
    r"sarl|sas|gmbh|ltd\.?|limited|inc\.?|incorporated|bv|sa|ag|plc|llc|corp\.?|corporation"
    r")\b\.?",
    re.IGNORECASE,
)

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")

_BLOCK_TAG_PATTERN = re.compile(
    r"<\s*/?\s*(p|div|br|li|h[1-6]|tr|table|section|article)\b[^>]*>",
    re.IGNORECASE,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def normalize_title(title: str) -> str:
    """Lowercase title with seniority markers and punctuation removed."""
    text = title.casefold()
    text = _SENIORITY_PATTERN.sub(" ", text)
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_company_name(name: str) -> str:
    """Lowercase company name with legal suffixes and punctuation removed."""
    text = name.casefold()
    text = _LEGAL_SUFFIX_PATTERN.sub(" ", text)
    text = _PUNCTUATION_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def fix_mojibake(text: str) -> str:
    """Repair UTF-8-decoded-as-Latin-1 mojibake (seen from some sources, e.g. RemoteOK).

    A no-op on already-correct text, so it is safe to apply unconditionally.
    """
    if not text:
        return text
    return ftfy.fix_text(text)


def clean_text(text: str | None) -> str | None:
    """Repair mojibake and trim whitespace on a plain (non-HTML) text field."""
    if text is None:
        return None
    cleaned = fix_mojibake(text).strip()
    return cleaned or None


def strip_html(description: str) -> str:
    """Convert HTML to plain text, preserving paragraph breaks as blank lines."""
    if not description:
        return ""

    text = _BLOCK_TAG_PATTERN.sub("\n\n", description)
    text = _TAG_PATTERN.sub(" ", text)

    parser = _HTMLTextExtractor()
    parser.feed(text)
    text = parser.get_text()
    text = html.unescape(text)
    text = fix_mojibake(text)
    text = unicodedata.normalize("NFKC", text)

    paragraphs = [_WHITESPACE_PATTERN.sub(" ", part).strip() for part in text.split("\n\n")]
    paragraphs = [part for part in paragraphs if part]
    return "\n\n".join(paragraphs)


def normalize_salary(
    salary_min: float | int | None,
    salary_max: float | int | None,
    currency: str | None = None,
    period: str | None = None,
) -> tuple[float | None, float | None, str | None, str | None]:
    """Normalize salary to (min, max, currency, period). Single values fill both bounds."""
    min_val = float(salary_min) if salary_min is not None else None
    max_val = float(salary_max) if salary_max is not None else None

    if min_val is not None and max_val is None:
        max_val = min_val
    elif max_val is not None and min_val is None:
        min_val = max_val

    currency_norm = currency.upper() if currency else None
    period_norm = period.lower() if period else None
    return min_val, max_val, currency_norm, period_norm
