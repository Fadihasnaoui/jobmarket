"""Integration tests for the parse runner's batch loop."""

from __future__ import annotations

import asyncio
import uuid

from sqlalchemy import delete, select

from jobmarket.db.models import Job, JobSource, RawJob
from jobmarket.db.session import async_session_factory
from jobmarket.parse.runner import run_parse


async def test_run_parse_terminates_when_only_permanently_failing_rows_remain(
    adzuna_payload: dict[str, object],
) -> None:
    """A row that can never parse (e.g. missing required field) must not loop forever.

    Regression test for a real bug: the runner's `while True` loop re-selected rows
    with parsed_at IS NULL on every pass, and a permanently-failing row never gets
    parsed_at set — so once only such rows remained, the loop never terminated. This
    was discovered live: a background `parse` run spun for hours and wrote 4.5GB of
    log output re-processing the same ~375 rows before being killed.
    """
    suffix = uuid.uuid4().hex[:8]
    broken_payload = {k: v for k, v in adzuna_payload.items() if k != "title"}
    source_id = f"pytest-broken-{suffix}"

    async with async_session_factory() as session:
        raw_job = RawJob(source="adzuna", source_id=source_id, payload=broken_payload)
        session.add(raw_job)
        await session.commit()
        await session.refresh(raw_job)
        raw_job_id = raw_job.id

    try:
        # If the infinite-loop bug regresses, this hangs until the timeout instead
        # of returning — that's the regression signal.
        summary = await asyncio.wait_for(run_parse(source="adzuna"), timeout=30)
        assert summary["failed_count"] >= 1

        async with async_session_factory() as session:
            raw_job = await session.get(RawJob, raw_job_id)
            assert raw_job is not None
            assert raw_job.parsed_at is None
    finally:
        async with async_session_factory() as session:
            await session.execute(
                delete(JobSource).where(
                    JobSource.raw_job_id.in_(
                        select(RawJob.id).where(RawJob.source_id == source_id)
                    )
                )
            )
            await session.execute(delete(RawJob).where(RawJob.source_id == source_id))
            await session.commit()


async def test_run_parse_recovers_unrelated_good_rows_alongside_a_broken_one(
    adzuna_payload: dict[str, object],
) -> None:
    """A permanently-broken row must not block good rows in the same batch/run."""
    suffix = uuid.uuid4().hex[:8]
    broken_payload = {k: v for k, v in adzuna_payload.items() if k != "title"}
    broken_source_id = f"pytest-broken2-{suffix}"
    good_payload = {**adzuna_payload, "id": f"pytest-good-{suffix}"}
    good_source_id = f"pytest-good-{suffix}"

    job_id: int | None = None
    try:
        async with async_session_factory() as session:
            broken = RawJob(source="adzuna", source_id=broken_source_id, payload=broken_payload)
            good = RawJob(source="adzuna", source_id=good_source_id, payload=good_payload)
            session.add_all([broken, good])
            await session.commit()
            await session.refresh(broken)
            await session.refresh(good)
            broken_id, good_id = broken.id, good.id

        await asyncio.wait_for(run_parse(source="adzuna"), timeout=30)

        async with async_session_factory() as session:
            broken_row = await session.get(RawJob, broken_id)
            good_row = await session.get(RawJob, good_id)
            assert broken_row is not None and broken_row.parsed_at is None
            assert good_row is not None and good_row.parsed_at is not None

            link = await session.scalar(
                select(JobSource.job_id).where(JobSource.raw_job_id == good_id)
            )
            job_id = link
    finally:
        async with async_session_factory() as session:
            await session.execute(
                delete(JobSource).where(
                    JobSource.raw_job_id.in_(
                        select(RawJob.id).where(
                            RawJob.source_id.in_([broken_source_id, good_source_id])
                        )
                    )
                )
            )
            await session.execute(
                delete(RawJob).where(RawJob.source_id.in_([broken_source_id, good_source_id]))
            )
            if job_id is not None:
                remaining = await session.scalar(
                    select(JobSource.job_id).where(JobSource.job_id == job_id)
                )
                if remaining is None:
                    await session.execute(delete(Job).where(Job.id == job_id))
            await session.commit()
