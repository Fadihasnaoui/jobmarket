"""Resumable batched job embedding runner."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from sqlalchemy import text

from jobmarket.db.session import async_session_factory
from jobmarket.embeddings.config import EmbeddingSettings, get_embedding_settings
from jobmarket.embeddings.encoder import (
    EmbeddingEncoder,
    SentenceTransformerEncoder,
    prepare_vector,
    vector_to_pg,
)
from jobmarket.embeddings.text import build_job_embedding_text


@dataclass(frozen=True)
class JobEmbeddingConfig:
    """Runtime options for embedding normalized jobs."""

    limit: int | None = None
    batch_size: int = 64
    re_embed: bool = False
    job_id: int | None = None
    dry_run: bool = False
    model: str | None = None
    device: str = "cpu"
    run_id: int = 342


@dataclass(frozen=True)
class JobEmbeddingSummary:
    """Observable embedding-run summary."""

    selected_jobs: int
    embedded_jobs: int
    pending_jobs: int
    failed_batches: int
    model_name: str
    embedding_version: str
    dimension: int
    dry_run: bool
    elapsed_seconds: float
    average_jobs_per_second: float


async def run_job_embedding(
    config: JobEmbeddingConfig,
    *,
    encoder: EmbeddingEncoder | None = None,
    settings: EmbeddingSettings | None = None,
) -> JobEmbeddingSummary:
    """Embed pending jobs in deterministic batches."""
    if config.limit is not None and config.limit <= 0:
        raise ValueError("limit must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    settings = settings or get_embedding_settings(
        model_name=config.model, batch_size=config.batch_size
    )
    if settings.dimension != 384:
        raise ValueError("Phase C expects 384-dimensional job embeddings")
    if encoder is None and not config.dry_run:
        encoder = SentenceTransformerEncoder(settings, device=config.device)
    started = perf_counter()
    selected = 0
    embedded = 0
    failed_batches = 0
    async with async_session_factory() as session:
        job_ids = await _select_pending_job_ids(session, config, settings)
    selected = len(job_ids)
    if config.dry_run or not job_ids:
        elapsed = perf_counter() - started
        return JobEmbeddingSummary(
            selected_jobs=selected,
            embedded_jobs=0,
            pending_jobs=selected,
            failed_batches=0,
            model_name=settings.model_name,
            embedding_version=settings.embedding_version,
            dimension=settings.dimension,
            dry_run=config.dry_run,
            elapsed_seconds=elapsed,
            average_jobs_per_second=0.0,
        )
    if encoder is None:
        raise RuntimeError("encoder is required unless dry_run is enabled")
    for batch_ids in _chunks(job_ids, settings.batch_size):
        try:
            async with async_session_factory() as session:
                rows = await _load_embedding_rows(session, batch_ids, config.run_id)
            texts = [build_job_embedding_text(row["job"], row["enrichment"]) for row in rows]
            vectors = encoder.encode(texts)
            if len(vectors) != len(rows):
                raise ValueError("encoder returned a different number of vectors than input texts")
            prepared = [
                prepare_vector(vector, dimension=settings.dimension, normalize=settings.normalize)
                for vector in vectors
            ]
            async with async_session_factory() as session:
                for row, vector in zip(rows, prepared, strict=True):
                    await _persist_embedding(session, int(row["job"]["id"]), vector, settings)
                await session.commit()
            embedded += len(rows)
        except Exception:
            failed_batches += 1
            continue
    elapsed = perf_counter() - started
    return JobEmbeddingSummary(
        selected_jobs=selected,
        embedded_jobs=embedded,
        pending_jobs=selected - embedded,
        failed_batches=failed_batches,
        model_name=settings.model_name,
        embedding_version=settings.embedding_version,
        dimension=settings.dimension,
        dry_run=False,
        elapsed_seconds=elapsed,
        average_jobs_per_second=embedded / elapsed if elapsed > 0 else 0.0,
    )


async def count_pending_jobs(config: JobEmbeddingConfig, settings: EmbeddingSettings) -> int:
    """Count jobs that would be selected by the embedding runner."""
    async with async_session_factory() as session:
        return len(await _select_pending_job_ids(session, config, settings))


async def _select_pending_job_ids(
    session: Any,
    config: JobEmbeddingConfig,
    settings: EmbeddingSettings,
) -> list[int]:
    if not await _embedding_columns_available(session):
        raise RuntimeError(
            "job embedding columns are missing; install pgvector and apply Alembic migration 005"
        )
    clauses = ["jer.run_id = :run_id", "jer.status = 'success'"]
    params: dict[str, Any] = {
        "run_id": config.run_id,
        "model": settings.model_name,
        "version": settings.embedding_version,
    }
    if config.job_id is not None:
        clauses.append("j.id = :job_id")
        params["job_id"] = config.job_id
    if not config.re_embed:
        clauses.append(
            "(j.embedding IS NULL OR j.embedding_model IS DISTINCT FROM :model "
            "OR j.embedding_version IS DISTINCT FROM :version)"
        )
    limit_sql = ""
    if config.limit is not None:
        limit_sql = " LIMIT :limit"
        params["limit"] = config.limit
    sql = text(
        "SELECT j.id FROM jobs j "
        "JOIN job_enrichment_results jer ON jer.job_id = j.id "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY j.id ASC" + limit_sql
    )
    return [int(row[0]) for row in (await session.execute(sql, params)).all()]


async def _embedding_columns_available(session: Any) -> bool:
    rows = await session.execute(
        text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'jobs'
              AND column_name IN (
                  'embedding', 'embedding_model', 'embedding_version', 'embedding_created_at'
              )
            """
        )
    )
    return {str(row[0]) for row in rows.all()} == {
        "embedding",
        "embedding_model",
        "embedding_version",
        "embedding_created_at",
    }


async def _load_embedding_rows(
    session: Any, job_ids: list[int], run_id: int
) -> list[dict[str, Any]]:
    sql = text(
        """
        SELECT
            j.id, j.title, j.description, j.country, j.city, j.contract_type, j.is_remote,
            COALESCE(
                array_agg(s.canonical ORDER BY s.canonical)
                FILTER (WHERE s.canonical IS NOT NULL),
                '{}'
            ) AS skills
        FROM jobs j
        JOIN job_enrichment_results jer ON jer.job_id = j.id AND jer.run_id = :run_id
        LEFT JOIN job_skills js ON js.job_id = j.id AND js.run_id = :run_id
        LEFT JOIN skills s ON s.id = js.skill_id
        WHERE j.id = ANY(:job_ids)
        GROUP BY j.id
        ORDER BY j.id ASC
        """
    )
    rows = (await session.execute(sql, {"job_ids": job_ids, "run_id": run_id})).mappings().all()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "job": {
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "country": row["country"],
                    "city": row["city"],
                    "contract_type": row["contract_type"],
                    "is_remote": row["is_remote"],
                },
                "enrichment": {"skills": list(row["skills"] or [])},
            }
        )
    return result


async def _persist_embedding(
    session: Any,
    job_id: int,
    vector: list[float],
    settings: EmbeddingSettings,
) -> None:
    await session.execute(
        text(
            """
            UPDATE jobs
            SET embedding = CAST(:embedding AS vector),
                embedding_model = :model,
                embedding_version = :version,
                embedding_created_at = now()
            WHERE id = :job_id
            """
        ),
        {
            "job_id": job_id,
            "embedding": vector_to_pg(vector),
            "model": settings.model_name,
            "version": settings.embedding_version,
        },
    )


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


__all__ = ["JobEmbeddingConfig", "JobEmbeddingSummary", "count_pending_jobs", "run_job_embedding"]
