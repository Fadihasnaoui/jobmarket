"""Integration test for re-parse from raw store."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import Job, JobSource, RawJob
from jobmarket.db.session import async_session_factory
from jobmarket.parse.adzuna import AdzunaParser
from jobmarket.parse.dedup import persist_parsed_job


async def test_reparse_from_raw_store_keeps_single_job(
    adzuna_payload: dict[str, object],
) -> None:
    """Simulates --reparse: clear parsed_at, then parse again from raw payload."""
    source_id = f"pytest-reparse-{uuid.uuid4().hex[:8]}"
    job_id: int | None = None

    try:
        async with async_session_factory() as session:
            assert isinstance(session, AsyncSession)
            raw_job = RawJob(source="adzuna", source_id=source_id, payload=adzuna_payload)
            session.add(raw_job)
            await session.commit()
            await session.refresh(raw_job)

            parsed = AdzunaParser().parse(adzuna_payload)
            job_id = await persist_parsed_job(session, raw_job_id=raw_job.id, parsed=parsed)
            raw_job.parsed_at = datetime.now(UTC)
            await session.commit()

        async with async_session_factory() as session:
            await session.execute(
                update(RawJob).where(RawJob.source_id == source_id).values(parsed_at=None)
            )
            await session.commit()

            raw_job = (
                await session.execute(select(RawJob).where(RawJob.source_id == source_id))
            ).scalar_one()
            assert raw_job.parsed_at is None

            parsed = AdzunaParser().parse(adzuna_payload)
            await persist_parsed_job(session, raw_job_id=raw_job.id, parsed=parsed)
            raw_job.parsed_at = datetime.now(UTC)
            await session.commit()

            unique_jobs = await session.scalar(
                select(func.count(func.distinct(JobSource.job_id))).where(
                    JobSource.raw_job_id == raw_job.id
                )
            )
            link_count = await session.scalar(
                select(func.count())
                .select_from(JobSource)
                .where(JobSource.raw_job_id == raw_job.id)
            )

        assert unique_jobs == 1
        assert link_count == 1
    finally:
        async with async_session_factory() as session:
            raw_row = await session.scalar(
                select(RawJob.id).where(RawJob.source_id == source_id)
            )
            if raw_row is not None:
                await session.execute(
                    delete(JobSource).where(JobSource.raw_job_id == raw_row)
                )
                await session.execute(delete(RawJob).where(RawJob.id == raw_row))
            if job_id is not None:
                remaining_links = await session.scalar(
                    select(func.count())
                    .select_from(JobSource)
                    .where(JobSource.job_id == job_id)
                )
                if remaining_links == 0:
                    await session.execute(delete(Job).where(Job.id == job_id))
            await session.commit()


async def test_reparse_updates_title_and_description_in_place(
    adzuna_payload: dict[str, object],
) -> None:
    """A parser fix (same job_hash) must overwrite stale title/description, not freeze them.

    Simulates content written by a since-fixed parser bug (e.g. the RemoteOK mojibake
    issue) by persisting a hand-built ParsedJob directly, bypassing the parser, then
    re-persisting the real parser's (corrected) output for the same raw_job_id.
    """
    source_id = f"pytest-reparse-content-{uuid.uuid4().hex[:8]}"
    job_id: int | None = None

    try:
        async with async_session_factory() as session:
            raw_job = RawJob(source="adzuna", source_id=source_id, payload=adzuna_payload)
            session.add(raw_job)
            await session.commit()
            await session.refresh(raw_job)

            stale = AdzunaParser().parse(adzuna_payload).model_copy(
                update={"description": "stale pre-fix description"}
            )
            job_id = await persist_parsed_job(session, raw_job_id=raw_job.id, parsed=stale)
            await session.commit()

            job = await session.get(Job, job_id)
            assert job is not None
            assert job.description == "stale pre-fix description"

            fixed_parsed = AdzunaParser().parse(adzuna_payload)
            await persist_parsed_job(session, raw_job_id=raw_job.id, parsed=fixed_parsed)
            await session.commit()

            await session.refresh(job)
            assert job.description == fixed_parsed.description
            assert job.description != "stale pre-fix description"
    finally:
        async with async_session_factory() as session:
            raw_row = await session.scalar(
                select(RawJob.id).where(RawJob.source_id == source_id)
            )
            if raw_row is not None:
                await session.execute(
                    delete(JobSource).where(JobSource.raw_job_id == raw_row)
                )
                await session.execute(delete(RawJob).where(RawJob.id == raw_row))
            if job_id is not None:
                remaining_links = await session.scalar(
                    select(func.count())
                    .select_from(JobSource)
                    .where(JobSource.job_id == job_id)
                )
                if remaining_links == 0:
                    await session.execute(delete(Job).where(Job.id == job_id))
            await session.commit()
