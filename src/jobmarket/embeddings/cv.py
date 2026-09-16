"""In-memory CV embedding helpers."""

from __future__ import annotations

from jobmarket.cv.profile import CvProfile
from jobmarket.embeddings.config import EmbeddingSettings, get_embedding_settings
from jobmarket.embeddings.encoder import (
    EmbeddingEncoder,
    SentenceTransformerEncoder,
    prepare_vector,
)
from jobmarket.embeddings.text import build_cv_embedding_text


def embed_cv_profile(
    profile: CvProfile,
    *,
    encoder: EmbeddingEncoder | None = None,
    settings: EmbeddingSettings | None = None,
    device: str = "cpu",
) -> list[float]:
    """Return one normalized CV embedding without persisting raw CV text."""
    settings = settings or get_embedding_settings()
    if settings.dimension != 384:
        raise ValueError("Phase C expects 384-dimensional CV embeddings")
    encoder = encoder or SentenceTransformerEncoder(settings, device=device)
    text = build_cv_embedding_text(profile)
    vectors = encoder.encode([text])
    if len(vectors) != 1:
        raise ValueError("encoder must return exactly one CV vector")
    return prepare_vector(vectors[0], dimension=settings.dimension, normalize=settings.normalize)


__all__ = ["embed_cv_profile"]
