"""Parse runner: normalize raw payloads into clean tables."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update

from jobmarket.db.models import RawJob
from jobmarket.db.session import async_session_factory
from jobmarket.parse.dedup import persist_parsed_job
from jobmarket.parse.registry import PARSER_REGISTRY, get_parser

logger = structlog.get_logger(__name__)

BATCH_SIZE = 500


async def run_parse(source: str | None = None, *, reparse: bool = False) -> dict[str, int]:
    """Parse unparsed raw jobs into clean tables."""
    parsed_count = 0
    failed_count = 0
    # Rows that fail this run are excluded from the next SELECT. Without this, a
    # permanently-malformed row (parsed_at never gets set) is reselected forever —
    # the `while True` loop would never terminate once only such rows remain.
    failed_ids: set[int] = set()

    async with async_session_factory() as session:
        if reparse:
            if source is None:
                raise ValueError("--reparse requires --source")
            await session.execute(
                update(RawJob).where(RawJob.source == source).values(parsed_at=None)
            )
            await session.commit()
            logger.info("parse_reparse_reset", source=source)

        while True:
            stmt = (
                select(RawJob).where(RawJob.parsed_at.is_(None)).order_by(RawJob.id).limit(BATCH_SIZE)
            )
            if failed_ids:
                stmt = stmt.where(RawJob.id.notin_(failed_ids))
            if source is not None:
                stmt = stmt.where(RawJob.source == source)
            else:
                stmt = stmt.where(RawJob.source.in_(supported_sources()))

            result = await session.execute(stmt)
            raw_jobs = list(result.scalars().all())
            if not raw_jobs:
                break

            for raw_job in raw_jobs:
                parser = get_parser(raw_job.source)
                try:
                    parsed = parser.parse(raw_job.payload)
                    await persist_parsed_job(
                        session,
                        raw_job_id=raw_job.id,
                        parsed=parsed,
                    )
                    raw_job.parsed_at = datetime.now(UTC)
                    parsed_count += 1
                except Exception as exc:
                    failed_count += 1
                    failed_ids.add(raw_job.id)
                    logger.warning(
                        "parse_row_failed",
                        raw_job_id=raw_job.id,
                        source=raw_job.source,
                        error=str(exc),
                    )

            await session.commit()

    logger.info(
        "parse_run_complete",
        source=source,
        parsed_count=parsed_count,
        failed_count=failed_count,
    )
    return {"parsed_count": parsed_count, "failed_count": failed_count}


def supported_sources() -> list[str]:
    """Sources that currently have a parser implementation."""
    return sorted(PARSER_REGISTRY)
