"""Embedding configuration and deterministic strategy metadata."""

from __future__ import annotations

from dataclasses import dataclass

from jobmarket.config import get_settings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_NORMALIZE = True
EMBEDDING_TEXT_STRATEGY_VERSION = "job-embedding-v1"


@dataclass(frozen=True)
class EmbeddingSettings:
    """Centralized embedding runtime settings."""

    model_name: str
    dimension: int
    batch_size: int
    normalize: bool
    model_revision: str | None
    strategy_version: str = EMBEDDING_TEXT_STRATEGY_VERSION

    @property
    def embedding_version(self) -> str:
        """Version stamp for stored job embeddings."""
        revision = self.model_revision or "default"
        norm = "l2" if self.normalize else "raw"
        return f"{self.strategy_version}:{self.model_name}:{revision}:{self.dimension}:{norm}"


def get_embedding_settings(
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
) -> EmbeddingSettings:
    """Return embedding settings from environment plus optional CLI overrides."""
    settings = get_settings()
    return EmbeddingSettings(
        model_name=model_name or settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=batch_size or settings.embedding_batch_size,
        normalize=settings.embedding_normalize,
        model_revision=settings.embedding_model_revision or None,
    )


__all__ = [
    "DEFAULT_EMBEDDING_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_TEXT_STRATEGY_VERSION",
    "EmbeddingSettings",
    "get_embedding_settings",
]
