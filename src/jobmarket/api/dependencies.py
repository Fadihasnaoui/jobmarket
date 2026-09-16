"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import OpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.api.store import CvStore
from jobmarket.db.session import async_session_factory
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.skills.ontology import SkillsOntology, load_ontology

_cv_store = CvStore()


def get_cv_store() -> CvStore:
    """Process-wide CV store singleton (override in tests via `dependency_overrides`)."""
    return _cv_store


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session


def get_ontology() -> SkillsOntology:
    return load_ontology()


def get_upload_encoder() -> EmbeddingEncoder | None:
    """Encoder used to embed a freshly uploaded CV.

    `None` means "build the real local sentence-transformers model" — the only model
    inference the API performs on any request path (see `store.CachedVectorEncoder`
    for how later `/matches`/`/skill-gap` calls reuse that one embedding instead of
    re-running the model). Tests override this to inject a fake encoder.
    """
    return None


def get_chat_llm_client() -> OpenAI | None:
    """LLM client used by the chat router's off-topic/rag classification and the RAG
    answerer's grounded-generation call.

    `None` means "build the real client lazily inside `chat/llm_client.py` when
    actually needed" — same `None`-means-real convention as `get_upload_encoder`.
    Tests override this to inject a mock client.
    """
    return None


__all__ = [
    "get_chat_llm_client",
    "get_cv_store",
    "get_db_session",
    "get_ontology",
    "get_upload_encoder",
]
