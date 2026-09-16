"""Integration tests for ingest idempotency."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import RawJob
from jobmarket.db.session import async_session_factory
from jobmarket.ingest.base import RawRecord
from jobmarket.ingest.runner import _insert_batch


async def test_insert_batch_is_idempotent() -> None:
    source = f"pytest-idempotency-{uuid.uuid4().hex[:8]}"
    records = [
        RawRecord(source_id="fixture-1", payload={"title": "Engineer"}),
        RawRecord(source_id="fixture-2", payload={"title": "Analyst"}),
    ]

    try:
        async with async_session_factory() as session:
            assert isinstance(session, AsyncSession)
            first_inserted = await _insert_batch(session, source, records)
            second_inserted = await _insert_batch(session, source, records)
            total = await session.scalar(
                select(func.count()).select_from(RawJob).where(RawJob.source == source)
            )

        assert first_inserted == 2
        assert second_inserted == 0
        assert total == 2
    finally:
        async with async_session_factory() as session:
            await session.execute(delete(RawJob).where(RawJob.source == source))
            await session.commit()
