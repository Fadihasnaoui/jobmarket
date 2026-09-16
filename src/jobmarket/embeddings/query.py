"""In-memory embedding helper for a raw chat question (not a CV or job posting)."""

from __future__ import annotations

from jobmarket.embeddings.config import EmbeddingSettings, get_embedding_settings
from jobmarket.embeddings.encoder import (
    EmbeddingEncoder,
    SentenceTransformerEncoder,
    prepare_vector,
)


def embed_query_text(
    text: str,
    *,
    encoder: EmbeddingEncoder | None = None,
    settings: EmbeddingSettings | None = None,
    device: str = "cpu",
) -> list[float]:
    """Return one normalized embedding for a raw text query.

    Same shape as `embeddings/cv.py::embed_cv_profile`, minus the CV-specific text
    building — a chat question is already the text to embed, not a structured object
    that needs formatting first.
    """
    settings = settings or get_embedding_settings()
    if settings.dimension != 384:
        raise ValueError("Phase C expects 384-dimensional embeddings")
    encoder = encoder or SentenceTransformerEncoder(settings, device=device)
    vectors = encoder.encode([text])
    if len(vectors) != 1:
        raise ValueError("encoder must return exactly one query vector")
    return prepare_vector(vectors[0], dimension=settings.dimension, normalize=settings.normalize)


__all__ = ["embed_query_text"]
