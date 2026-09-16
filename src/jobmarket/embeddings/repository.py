"""Data access helpers for job embeddings and semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.embeddings.config import EmbeddingSettings
from jobmarket.embeddings.encoder import vector_to_pg


@dataclass(frozen=True)
class SemanticSearchFilters:
    """Optional filters for pgvector nearest-neighbour retrieval."""

    country: str | None = None
    career_level: str | None = None
    contract_type: str | None = None
    work_mode: str | None = None


@dataclass(frozen=True)
class SemanticJobHit:
    """One semantic nearest-neighbour hit."""

    job_id: int
    title: str
    company: str
    country: str | None
    city: str | None
    contract_type: str | None
    is_remote: bool | None
    cosine_distance: float
    semantic_score: float
    semantic_rank: int


async def search_jobs_by_embedding(
    session: AsyncSession,
    query_embedding: list[float],
    *,
    limit: int,
    filters: SemanticSearchFilters | None = None,
) -> list[SemanticJobHit]:
    """Search embedded jobs by cosine distance with deterministic tie-breaking."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    filters = filters or SemanticSearchFilters()
    clauses = ["j.embedding IS NOT NULL"]
    params: dict[str, Any] = {"embedding": vector_to_pg(query_embedding), "limit": limit}
    if filters.country:
        clauses.append("j.country = :country")
        params["country"] = filters.country
    if filters.contract_type:
        clauses.append("j.contract_type = :contract_type")
        params["contract_type"] = filters.contract_type
    if filters.work_mode == "remote":
        clauses.append("j.is_remote IS TRUE")
    elif filters.work_mode == "on_site":
        clauses.append("j.is_remote IS FALSE")
    elif filters.work_mode == "hybrid":
        clauses.append("(j.description ILIKE '%hybrid%' OR j.description ILIKE '%hybride%')")
    if filters.career_level:
        pattern = _career_level_pattern(filters.career_level)
        if pattern:
            clauses.append("(j.title ~* :career_pattern OR j.description ~* :career_pattern)")
            params["career_pattern"] = pattern
    where_sql = " AND ".join(clauses)
    sql = text(
        f"""
        SELECT
            j.id AS job_id,
            j.title,
            c.name AS company,
            j.country,
            j.city,
            j.contract_type,
            j.is_remote,
            (j.embedding <=> CAST(:embedding AS vector)) AS cosine_distance
        FROM jobs j
        JOIN companies c ON c.id = j.company_id
        WHERE {where_sql}
        ORDER BY cosine_distance ASC, j.id ASC
        LIMIT :limit
        """
    )
    rows = (await session.execute(sql, params)).mappings().all()
    hits: list[SemanticJobHit] = []
    for rank, row in enumerate(rows, start=1):
        distance = float(row["cosine_distance"])
        hits.append(
            SemanticJobHit(
                job_id=int(row["job_id"]),
                title=str(row["title"]),
                company=str(row["company"]),
                country=row["country"],
                city=row["city"],
                contract_type=row["contract_type"],
                is_remote=row["is_remote"],
                cosine_distance=distance,
                semantic_score=max(0.0, min(1.0, 1.0 - distance)),
                semantic_rank=rank,
            )
        )
    return hits


def current_embedding_clause(settings: EmbeddingSettings) -> str:
    """SQL predicate for jobs with current embedding metadata."""
    return "embedding IS NOT NULL AND embedding_model = :model AND embedding_version = :version"


def _career_level_pattern(level: str) -> str | None:
    patterns = {
        "junior": r"\\mjunior\\M|\\mdebutant\\M",
        "senior": r"\\msenior\\M|\\msr\\M|\\mconfirme\\M|\\mexperimente\\M",
        "lead": r"\\mlead\\M|tech lead|leader technique",
        "manager": r"\\mmanager\\M|responsable|head of|chef de projet",
        "internship": r"\\mstage\\M|stagiaire|internship|intern",
        "apprenticeship": r"alternance|apprenti|apprentice",
    }
    return patterns.get(level)


__all__ = ["SemanticJobHit", "SemanticSearchFilters", "search_jobs_by_embedding"]
