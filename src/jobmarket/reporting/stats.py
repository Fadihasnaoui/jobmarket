"""Database statistics for the stats CLI command."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import Base, Company, IngestRun, Job, JobSource, RawJob
from jobmarket.db.session import async_session_factory


@dataclass
class SourceStats:
    source: str
    raw_total: int
    parsed: int
    unparsed: int


@dataclass
class StatsReport:
    raw_jobs: int
    jobs: int
    companies: int
    job_sources: int
    ingest_runs: int
    salary_coverage_pct: float
    jobs_with_salary: int
    predicted_salary_count: int
    per_source: list[SourceStats] = field(default_factory=list)


async def collect_stats() -> StatsReport:
    """Gather pipeline statistics from the database."""
    async with async_session_factory() as session:
        raw_jobs = await _scalar_count(session, RawJob)
        jobs = await _scalar_count(session, Job)
        companies = await _scalar_count(session, Company)
        job_sources = await _scalar_count(session, JobSource)
        ingest_runs = await _scalar_count(session, IngestRun)

        jobs_with_salary = await session.scalar(
            select(func.count())
            .select_from(Job)
            .where(or_(Job.salary_min.isnot(None), Job.salary_max.isnot(None)))
        )
        predicted_salary_count = await session.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.salary_is_predicted.is_(True))
        )

        source_rows = await session.execute(
            select(
                RawJob.source,
                func.count().label("raw_total"),
                func.count(RawJob.parsed_at).label("parsed"),
            ).group_by(RawJob.source)
        )

        per_source = [
            SourceStats(
                source=row.source,
                raw_total=row.raw_total,
                parsed=row.parsed,
                unparsed=row.raw_total - row.parsed,
            )
            for row in source_rows
        ]
        per_source.sort(key=lambda item: item.source)

    salary_count = jobs_with_salary or 0
    coverage = (salary_count / jobs * 100) if jobs else 0.0

    return StatsReport(
        raw_jobs=raw_jobs,
        jobs=jobs,
        companies=companies,
        job_sources=job_sources,
        ingest_runs=ingest_runs,
        salary_coverage_pct=coverage,
        jobs_with_salary=salary_count,
        predicted_salary_count=predicted_salary_count or 0,
        per_source=per_source,
    )


def format_stats(report: StatsReport) -> str:
    """Render a human-readable stats report."""
    lines = [
        "Pipeline statistics",
        "===================",
        f"raw_jobs:      {report.raw_jobs:,}",
        f"jobs:          {report.jobs:,}",
        f"companies:     {report.companies:,}",
        f"job_sources:   {report.job_sources:,}",
        f"ingest_runs:   {report.ingest_runs:,}",
        "",
        "Salary coverage",
        "---------------",
        f"jobs with salary:     {report.jobs_with_salary:,} ({report.salary_coverage_pct:.1f}%)",
        f"predicted salaries:   {report.predicted_salary_count:,}",
        "",
        "Per source",
        "----------",
    ]

    if not report.per_source:
        lines.append("(no raw rows yet)")
    else:
        lines.append(f"{'source':<12} {'raw':>8} {'parsed':>8} {'unparsed':>10}")
        for item in report.per_source:
            lines.append(
                f"{item.source:<12} {item.raw_total:>8,} {item.parsed:>8,} {item.unparsed:>10,}"
            )

    return "\n".join(lines)


async def _scalar_count(session: AsyncSession, model: type[Base]) -> int:
    value = await session.scalar(select(func.count()).select_from(model))
    return value or 0
