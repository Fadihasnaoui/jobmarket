"""Integration tests for cross-source deduplication."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select

from jobmarket.db.models import Job, JobSource, RawJob
from jobmarket.db.session import async_session_factory
from jobmarket.parse.adzuna import AdzunaParser
from jobmarket.parse.dedup import persist_parsed_job
from jobmarket.parse.remoteok import RemoteOKParser


async def test_same_logical_job_from_two_sources_dedupes(
    adzuna_payload: dict[str, object],
    remoteok_payload: dict[str, object],
) -> None:
    suffix = uuid.uuid4().hex[:8]
    adzuna_source_id = f"dedup-adzuna-{suffix}"
    remoteok_source_id = f"dedup-remoteok-{suffix}"
    job_id: int | None = None

    remoteok_payload = {
        **remoteok_payload,
        "position": adzuna_payload["title"],
        "company": "Acme Analytics SARL",
        "location": "Paris, France",
    }

    try:
        async with async_session_factory() as session:
            raw_adzuna = RawJob(
                source="adzuna", source_id=adzuna_source_id, payload=adzuna_payload
            )
            raw_remoteok = RawJob(
                source="remoteok",
                source_id=remoteok_source_id,
                payload=remoteok_payload,
            )
            session.add_all([raw_adzuna, raw_remoteok])
            await session.commit()
            await session.refresh(raw_adzuna)
            await session.refresh(raw_remoteok)

            adzuna_parsed = AdzunaParser().parse(adzuna_payload)
            remoteok_parsed = RemoteOKParser().parse(remoteok_payload)
            remoteok_parsed = remoteok_parsed.model_copy(
                update={
                    "country": adzuna_parsed.country,
                    "city": adzuna_parsed.city,
                }
            )

            await persist_parsed_job(session, raw_job_id=raw_adzuna.id, parsed=adzuna_parsed)
            job_id = await persist_parsed_job(
                session, raw_job_id=raw_remoteok.id, parsed=remoteok_parsed
            )
            await session.commit()

            raw_ids = [raw_adzuna.id, raw_remoteok.id]
            unique_jobs = await session.scalar(
                select(func.count(func.distinct(JobSource.job_id))).where(
                    JobSource.raw_job_id.in_(raw_ids)
                )
            )
            source_links = await session.scalar(
                select(func.count()).select_from(JobSource).where(
                    JobSource.raw_job_id.in_(raw_ids)
                )
            )

        assert unique_jobs == 1
        assert source_links == 2
    finally:
        async with async_session_factory() as session:
            raw_ids = list(
                (
                    await session.execute(
                        select(RawJob.id).where(
                            RawJob.source_id.in_([adzuna_source_id, remoteok_source_id])
                        )
                    )
                ).scalars()
            )
            if raw_ids:
                await session.execute(
                    delete(JobSource).where(JobSource.raw_job_id.in_(raw_ids))
                )
                await session.execute(delete(RawJob).where(RawJob.id.in_(raw_ids)))
            if job_id is not None:
                remaining = await session.scalar(
                    select(func.count())
                    .select_from(JobSource)
                    .where(JobSource.job_id == job_id)
                )
                if remaining == 0:
                    await session.execute(delete(Job).where(Job.id == job_id))
            await session.commit()
