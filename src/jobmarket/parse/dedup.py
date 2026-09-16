"""Job deduplication and clean-table upsert logic."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import Company, Job, JobSource
from jobmarket.parse.base import ParsedJob
from jobmarket.parse.normalize import normalize_company_name, normalize_title

ENRICHMENT_COLUMNS = (
    "location_raw",
    "country",
    "city",
    "is_remote",
    "contract_type",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "salary_is_predicted",
    "posted_at",
    "url",
)


def compute_job_hash(title_norm: str, company_name_norm: str, country: str | None) -> str:
    """Stable hash for cross-source job deduplication."""
    country_part = country or ""
    digest_input = f"{title_norm}|{company_name_norm}|{country_part}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


async def persist_parsed_job(
    session: AsyncSession,
    *,
    raw_job_id: int,
    parsed: ParsedJob,
) -> int:
    """Upsert company/job rows and link the raw record. Returns the job id."""
    title_norm = normalize_title(parsed.title)
    company_name_norm = normalize_company_name(parsed.company_name)
    job_hash = compute_job_hash(title_norm, company_name_norm, parsed.country)

    company_id = await _get_or_create_company(session, parsed.company_name, company_name_norm)

    job_values = {
        "job_hash": job_hash,
        "company_id": company_id,
        "title": parsed.title,
        "title_norm": title_norm,
        "description": parsed.description,
        "location_raw": parsed.location_raw,
        "country": parsed.country,
        "city": parsed.city,
        "is_remote": parsed.is_remote,
        "contract_type": parsed.contract_type,
        "salary_min": parsed.salary_min,
        "salary_max": parsed.salary_max,
        "salary_currency": parsed.salary_currency,
        "salary_period": parsed.salary_period,
        "salary_is_predicted": parsed.salary_is_predicted,
        "posted_at": parsed.posted_at,
        "url": parsed.url,
    }

    insert_stmt = insert(Job).values(**job_values)
    update_set: dict[str, object] = {
        column: func.coalesce(getattr(Job, column), getattr(insert_stmt.excluded, column))
        for column in ENRICHMENT_COLUMNS
    }
    # Content fields always take the freshly parsed value (not coalesced) so that
    # `--reparse` actually propagates parser fixes to existing rows.
    update_set["title"] = insert_stmt.excluded.title
    update_set["title_norm"] = insert_stmt.excluded.title_norm
    update_set["description"] = insert_stmt.excluded.description
    update_set["updated_at"] = datetime.now(UTC)

    upsert_stmt = insert_stmt.on_conflict_do_update(
        index_elements=[Job.job_hash],
        set_=update_set,
    ).returning(Job.id)

    result = await session.execute(upsert_stmt)
    job_id = result.scalar_one()

    source_stmt = (
        insert(JobSource)
        .values(job_id=job_id, raw_job_id=raw_job_id)
        .on_conflict_do_nothing(index_elements=["job_id", "raw_job_id"])
    )
    await session.execute(source_stmt)
    return job_id


async def _get_or_create_company(
    session: AsyncSession,
    name: str,
    name_norm: str,
) -> int:
    insert_stmt = (
        insert(Company)
        .values(name=name, name_norm=name_norm)
        .on_conflict_do_nothing(index_elements=[Company.name_norm])
        .returning(Company.id)
    )
    result = await session.execute(insert_stmt)
    company_id = result.scalar_one_or_none()
    if company_id is not None:
        return company_id

    existing = await session.execute(select(Company.id).where(Company.name_norm == name_norm))
    return existing.scalar_one()
