"""In-memory, short-TTL CV profile store — no raw CV text persistence.

Uploaded CVs are parsed, extracted, and embedded once; the resulting profile (with
raw document text stripped — nothing downstream needs it again) plus the precomputed
CV embedding vector are cached here under a short-lived id so `/cv/{id}/matches` and
`/cv/{id}/skill-gap` never re-parse, re-extract, or re-embed on every request. Entries
expire on their own (lazy eviction on lookup); nothing here is ever written to disk or
a database, and the process restarting clears everything.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from jobmarket.cv.improver import ImprovedCvResult
from jobmarket.cv.profile import CvProfile
from jobmarket.cv.quality_report import ContactSignals

DEFAULT_TTL_SECONDS = 30 * 60  # 30 minutes


def redact_document_text(profile: CvProfile) -> CvProfile:
    """Strip raw CV text from a profile before caching it.

    Every fact needed downstream (skills, experience/education evidence, extraction
    quality) already lives on the profile's own structured fields — nothing reads
    `document.text`/`document.pages`/`document.blocks` again after extraction.
    """
    document = profile.document
    redacted_pages = [page.model_copy(update={"text": ""}) for page in document.pages]
    redacted_blocks = [block.model_copy(update={"text": ""}) for block in document.blocks]
    redacted_document = document.model_copy(
        update={"text": "", "pages": redacted_pages, "blocks": redacted_blocks}
    )
    return profile.model_copy(update={"document": redacted_document})


@dataclass(frozen=True)
class StoredCv:
    """One cached, privacy-safe CV profile plus its precomputed embedding.

    `improved_result` caches the one real LLM rewording call from `/cv/{id}/improve`
    so repeated calls (e.g. regenerating the .docx with a different name/contact)
    don't re-trigger it — same rationale as `cv_embedding` avoiding repeat model
    inference. It never carries name/contact: those are session-only, per-request
    values `docx_export.render_improved_cv_docx` takes directly and never touches
    this store.

    `contact_signals` is computed once, at upload time, by pattern-scanning the CV's
    raw document text for email/phone/LinkedIn-or-GitHub *shapes* — three booleans,
    never the matched substrings — before that raw text is redacted below. It exists
    because nothing later has access to the raw text to check against: without it,
    the CV quality report's "missing contact" check could only ever see the optional,
    caller-supplied name/contact form fields, not the CV's own real content.
    """

    profile: CvProfile
    cv_embedding: list[float] | None
    extraction_method: str
    fallback_reason: str | None
    out_of_scope_message: str | None
    filename: str
    created_at: datetime
    expires_at: datetime
    improved_result: ImprovedCvResult | None = None
    contact_signals: ContactSignals | None = None


class CachedVectorEncoder:
    """`EmbeddingEncoder` adapter that replays a precomputed vector.

    `cv/matching.py::recommend_jobs_for_cv` always calls `embed_cv_profile` itself for
    hybrid/semantic mode; passing this as its `encoder` makes that call reuse the
    vector computed once at upload time instead of re-running the local
    sentence-transformer model on every `/matches`/`/skill-gap` request. No network
    call either way — this only avoids repeat local model inference.
    """

    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = list(vector)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(self._vector) for _ in texts]


class CvStore:
    """Thread-safe, short-TTL in-memory store keyed by a random `cv_id`."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._items: dict[str, StoredCv] = {}
        self._lock = threading.Lock()

    def put(self, item: StoredCv) -> str:
        """Store a CV, enforcing the no-raw-text invariant regardless of the caller."""
        safe_item = replace(item, profile=redact_document_text(item.profile))
        cv_id = uuid4().hex
        with self._lock:
            self._items[cv_id] = safe_item
        return cv_id

    def get(self, cv_id: str) -> StoredCv | None:
        with self._lock:
            item = self._items.get(cv_id)
            if item is None:
                return None
            if item.expires_at <= datetime.now(UTC):
                del self._items[cv_id]
                return None
            return item

    def make_expiry(self) -> datetime:
        return datetime.now(UTC) + timedelta(seconds=self._ttl_seconds)

    def set_improved_result(self, cv_id: str, result: ImprovedCvResult) -> bool:
        """Cache a computed `ImprovedCvResult` under an existing cv_id.

        Returns False (no-op) if the id is unknown or has already expired — the
        caller should treat that the same as any other "unknown cv_id" case.
        """
        with self._lock:
            item = self._items.get(cv_id)
            if item is None or item.expires_at <= datetime.now(UTC):
                return False
            self._items[cv_id] = replace(item, improved_result=result)
            return True


__all__ = ["CachedVectorEncoder", "CvStore", "StoredCv", "redact_document_text"]
