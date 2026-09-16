"""Ingest runner: fetch raw records and persist to raw_jobs."""

from __future__ import annotations

import traceback
from datetime import UTC, datetime

import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import IngestRun, RawJob
from jobmarket.db.session import async_session_factory
from jobmarket.ingest.base import AbstractSource, RawRecord

logger = structlog.get_logger(__name__)

BATCH_SIZE = 500


async def run_ingest(source: AbstractSource) -> IngestRun:
    """Run ingest for a source, tracking progress in ingest_runs."""
    async with async_session_factory() as session:
        run = IngestRun(
            source=source.name,
            started_at=datetime.now(UTC),
            status="running",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        rows_fetched = 0
        rows_inserted = 0
        batch: list[RawRecord] = []

        try:
            async for record in source.fetch():
                rows_fetched += 1
                batch.append(record)

                if len(batch) >= BATCH_SIZE:
                    rows_inserted += await _insert_batch(session, source.name, batch)
                    batch.clear()

            if batch:
                rows_inserted += await _insert_batch(session, source.name, batch)

            run.status = "success"
            run.finished_at = datetime.now(UTC)
            run.rows_fetched = rows_fetched
            run.rows_inserted = rows_inserted
            await session.commit()

            logger.info(
                "ingest_run_complete",
                source=source.name,
                run_id=run.id,
                status=run.status,
                rows_fetched=rows_fetched,
                rows_inserted=rows_inserted,
            )
            return run

        except Exception:
            run.status = "failed"
            run.finished_at = datetime.now(UTC)
            run.rows_fetched = rows_fetched
            run.rows_inserted = rows_inserted
            run.error = traceback.format_exc()
            await session.commit()

            logger.error(
                "ingest_run_failed",
                source=source.name,
                run_id=run.id,
                rows_fetched=rows_fetched,
                rows_inserted=rows_inserted,
                error=run.error,
            )
            raise


async def _insert_batch(
    session: AsyncSession,
    source: str,
    records: list[RawRecord],
    *,
    commit: bool = True,
) -> int:
    """Insert a batch of raw records, skipping conflicts. Returns new row count."""
    if not records:
        return 0

    values = [
        {
            "source": source,
            "source_id": record.source_id,
            "payload": record.payload,
        }
        for record in records
    ]

    stmt = (
        insert(RawJob)
        .values(values)
        .on_conflict_do_nothing(constraint="uq_raw_jobs_source_source_id")
        .returning(RawJob.id)
    )
    result = await session.execute(stmt)
    if commit:
        await session.commit()
    else:
        await session.flush()
    return len(result.scalars().all())
