"""Embedding vector validation and encoding abstractions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from jobmarket.embeddings.config import EmbeddingSettings


class EmbeddingEncoder(Protocol):
    """Minimal encoder interface used by runners and CV matching."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode text strings into vectors."""


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """Return an L2-normalized copy of a finite vector."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise ValueError("embedding vector norm must be non-zero")
    return [float(value) / norm for value in vector]


def validate_vector(vector: Sequence[float], *, dimension: int) -> list[float]:
    """Validate dimension and numeric safety for one embedding vector."""
    values = [float(value) for value in vector]
    if len(values) != dimension:
        raise ValueError(f"embedding dimension {len(values)} does not match expected {dimension}")
    if any(math.isnan(value) or math.isinf(value) for value in values):
        raise ValueError("embedding vector contains NaN or infinite values")
    return values


def prepare_vector(
    vector: Sequence[float],
    *,
    dimension: int,
    normalize: bool,
) -> list[float]:
    """Validate and optionally L2-normalize one embedding vector."""
    values = validate_vector(vector, dimension=dimension)
    if normalize:
        values = normalize_vector(values)
    return validate_vector(values, dimension=dimension)


def vector_to_pg(value: Sequence[float]) -> str:
    """Serialize a vector for pgvector CAST(:value AS vector)."""
    return "[" + ",".join(f"{float(item):.9g}" for item in value) + "]"


class SentenceTransformerEncoder:
    """Lazy sentence-transformers encoder wrapper."""

    def __init__(self, settings: EmbeddingSettings, *, device: str = "cpu") -> None:
        self.settings = settings
        self.device = device
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional local install
            raise RuntimeError(
                "sentence-transformers is required for real embedding runs"
            ) from exc
        kwargs: dict[str, object] = {"device": device, "local_files_only": True}
        if settings.model_revision:
            kwargs["revision"] = settings.model_revision
        self._model = SentenceTransformer(settings.model_name, **kwargs)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts without persisting the source text."""
        vectors = self._model.encode(
            list(texts),
            batch_size=self.settings.batch_size,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return [[float(value) for value in row] for row in vectors]


__all__ = [
    "EmbeddingEncoder",
    "SentenceTransformerEncoder",
    "normalize_vector",
    "prepare_vector",
    "validate_vector",
    "vector_to_pg",
]
