"""Deduplication report for the dedup-report CLI command."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, select

from jobmarket.db.models import Job, JobSource, RawJob
from jobmarket.db.session import async_session_factory


@dataclass
class SourceDedupStats:
    source: str
    raw_rows: int
    linked_raw_rows: int
    unique_jobs: int
    source_duplication_factor: float


@dataclass
class DedupReport:
    total_raw_rows: int
    total_parsed_raw_rows: int
    total_unique_jobs: int
    linked_raw_rows: int
    duplication_factor: float
    per_source: list[SourceDedupStats] = field(default_factory=list)


async def collect_dedup_report() -> DedupReport:
    """Gather deduplication metrics from the database."""
    async with async_session_factory() as session:
        total_raw_rows = await session.scalar(select(func.count()).select_from(RawJob)) or 0
        total_parsed_raw_rows = await session.scalar(
            select(func.count())
            .select_from(RawJob)
            .where(RawJob.parsed_at.isnot(None))
        ) or 0
        total_unique_jobs = await session.scalar(select(func.count()).select_from(Job)) or 0
        linked_raw_rows = await session.scalar(select(func.count()).select_from(JobSource)) or 0

        source_rows = await session.execute(
            select(
                RawJob.source,
                func.count(RawJob.id).label("raw_rows"),
                func.count(JobSource.raw_job_id).label("linked_raw_rows"),
                func.count(func.distinct(JobSource.job_id)).label("unique_jobs"),
            )
            .select_from(RawJob)
            .outerjoin(JobSource, JobSource.raw_job_id == RawJob.id)
            .group_by(RawJob.source)
        )

        per_source: list[SourceDedupStats] = []
        for row in source_rows:
            unique_jobs = row.unique_jobs or 0
            linked = row.linked_raw_rows or 0
            factor = linked / unique_jobs if unique_jobs else 0.0
            per_source.append(
                SourceDedupStats(
                    source=row.source,
                    raw_rows=row.raw_rows,
                    linked_raw_rows=linked,
                    unique_jobs=unique_jobs,
                    source_duplication_factor=factor,
                )
            )
        per_source.sort(key=lambda item: item.source)

    duplication_factor = linked_raw_rows / total_unique_jobs if total_unique_jobs else 0.0

    return DedupReport(
        total_raw_rows=total_raw_rows,
        total_parsed_raw_rows=total_parsed_raw_rows,
        total_unique_jobs=total_unique_jobs,
        linked_raw_rows=linked_raw_rows,
        duplication_factor=duplication_factor,
        per_source=per_source,
    )


def format_dedup_report(report: DedupReport) -> str:
    """Render a human-readable deduplication report."""
    lines = [
        "Deduplication report",
        "====================",
        f"total raw rows:           {report.total_raw_rows:,}",
        f"parsed raw rows:          {report.total_parsed_raw_rows:,}",
        f"linked raw rows:          {report.linked_raw_rows:,}",
        f"total unique jobs:        {report.total_unique_jobs:,}",
        f"duplication factor:       {report.duplication_factor:.2f}x",
        "",
        "Per source",
        "----------",
    ]

    if not report.per_source:
        lines.append("(no raw rows yet)")
    else:
        lines.append(
            f"{'source':<12} {'raw':>8} {'linked':>8} {'unique':>8} {'factor':>8}"
        )
        for item in report.per_source:
            lines.append(
                f"{item.source:<12} {item.raw_rows:>8,} "
                f"{item.linked_raw_rows:>8,} {item.unique_jobs:>8,} "
                f"{item.source_duplication_factor:>7.2f}x"
            )

    lines.extend(
        [
            "",
            "Notes",
            "-----",
            "duplication factor = linked raw rows / unique jobs",
            "Values above 1.0 mean the same logical job appears on multiple boards.",
        ]
    )
    return "\n".join(lines)
