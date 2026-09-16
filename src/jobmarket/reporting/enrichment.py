"""Read-only reporting for persisted enrichment runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import mean, median
from typing import Any

from sqlalchemy import func, select

from jobmarket.db.models import (
    EnrichmentFailure,
    EnrichmentRun,
    Job,
    JobEnrichmentResult,
    JobSkill,
    JobSource,
    RawJob,
    Skill,
)
from jobmarket.db.session import async_session_factory


@dataclass(frozen=True)
class EnrichmentBreakdownRow:
    key: str
    jobs_processed: int
    jobs_succeeded: int
    jobs_failed: int
    jobs_with_skills: int
    zero_skill_jobs: int


@dataclass(frozen=True)
class TopSkillRow:
    canonical: str
    skill_rows: int
    jobs: int


@dataclass(frozen=True)
class FailureStageRow:
    stage: str
    failures: int
    attempts: int


@dataclass(frozen=True)
class EnrichmentRunReport:
    run_id: int
    run_type: str
    status: str
    matcher_version: str
    ontology_version: str
    config: dict[str, Any]
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None
    error: str | None
    jobs_total: int
    jobs_processed: int
    jobs_succeeded: int
    jobs_failed: int
    jobs_with_skills: int
    zero_skill_jobs: int
    persisted_skill_rows: int
    mean_skills_per_successful_job: float
    median_skills_per_successful_job: float
    mean_skills_among_jobs_with_skills: float
    zero_skill_rate: float
    failure_rate: float
    top_skills: list[TopSkillRow] = field(default_factory=list)
    source_breakdown: list[EnrichmentBreakdownRow] = field(default_factory=list)
    country_breakdown: list[EnrichmentBreakdownRow] = field(default_factory=list)
    failure_stage_breakdown: list[FailureStageRow] = field(default_factory=list)


async def collect_enrichment_report(run_id: int, *, top_n: int = 20) -> EnrichmentRunReport:
    """Collect a deterministic, read-only report for one enrichment run."""
    async with async_session_factory() as session:
        run = await session.get(EnrichmentRun, run_id)
        if run is None:
            raise ValueError(f"enrichment run {run_id} does not exist")

        skill_counts = list(
            await session.scalars(
                select(JobEnrichmentResult.skill_count)
                .where(
                    JobEnrichmentResult.run_id == run_id,
                    JobEnrichmentResult.status == "success",
                )
                .order_by(JobEnrichmentResult.job_id)
            )
        )
        positive_counts = [count for count in skill_counts if count > 0]
        persisted_skill_rows = await _scalar_int(
            select(func.count()).select_from(JobSkill).where(JobSkill.run_id == run_id)
        )
        top_skills = await _top_skills(run_id, top_n)
        source_breakdown = await _source_breakdown(run_id)
        country_breakdown = await _country_breakdown(run_id)
        failure_stage_breakdown = await _failure_stage_breakdown(run_id)

    succeeded = run.jobs_succeeded
    processed = run.jobs_processed
    return EnrichmentRunReport(
        run_id=run.id,
        run_type=run.run_type,
        status=run.status,
        matcher_version=run.matcher_version,
        ontology_version=run.ontology_version,
        config=run.config,
        started_at=run.started_at,
        updated_at=run.updated_at,
        finished_at=run.finished_at,
        error=run.error,
        jobs_total=run.jobs_total,
        jobs_processed=processed,
        jobs_succeeded=succeeded,
        jobs_failed=run.jobs_failed,
        jobs_with_skills=run.jobs_with_skills,
        zero_skill_jobs=run.zero_skill_jobs,
        persisted_skill_rows=persisted_skill_rows,
        mean_skills_per_successful_job=float(mean(skill_counts)) if skill_counts else 0.0,
        median_skills_per_successful_job=float(median(skill_counts)) if skill_counts else 0.0,
        mean_skills_among_jobs_with_skills=(
            float(mean(positive_counts)) if positive_counts else 0.0
        ),
        zero_skill_rate=(run.zero_skill_jobs / succeeded) if succeeded else 0.0,
        failure_rate=(run.jobs_failed / processed) if processed else 0.0,
        top_skills=top_skills,
        source_breakdown=source_breakdown,
        country_breakdown=country_breakdown,
        failure_stage_breakdown=failure_stage_breakdown,
    )


def format_enrichment_report(report: EnrichmentRunReport) -> str:
    """Render a human-readable enrichment report."""
    lines = [
        "Enrichment run report",
        "=====================",
        f"run id:                 {report.run_id}",
        f"run type:               {report.run_type}",
        f"status:                 {report.status}",
        f"matcher version:        {report.matcher_version}",
        f"ontology version:       {report.ontology_version}",
        f"config:                 {json.dumps(report.config, sort_keys=True)}",
        f"started at:             {report.started_at}",
        f"updated at:             {report.updated_at}",
        f"finished at:            {report.finished_at}",
        f"error:                  {report.error or ''}",
        "",
        "Counters",
        "--------",
        f"jobs total:             {report.jobs_total:,}",
        f"jobs processed:         {report.jobs_processed:,}",
        f"jobs succeeded:         {report.jobs_succeeded:,}",
        f"jobs failed:            {report.jobs_failed:,}",
        f"jobs with skills:       {report.jobs_with_skills:,}",
        f"zero-skill jobs:        {report.zero_skill_jobs:,}",
        f"persisted skill rows:   {report.persisted_skill_rows:,}",
        "",
        "Rates and averages",
        "------------------",
        f"mean skills/success:    {report.mean_skills_per_successful_job:.2f}",
        f"median skills/success:  {report.median_skills_per_successful_job:.2f}",
        f"mean skills/nonzero:    {report.mean_skills_among_jobs_with_skills:.2f}",
        f"zero-skill rate:        {report.zero_skill_rate:.1%}",
        f"failure rate:           {report.failure_rate:.1%}",
        "",
    ]
    _append_top_skills(lines, report.top_skills)
    _append_breakdown(lines, "Source breakdown", report.source_breakdown)
    _append_breakdown(lines, "Country breakdown", report.country_breakdown)
    _append_failure_stages(lines, report.failure_stage_breakdown)
    return "\n".join(lines)


def enrichment_report_to_json(report: EnrichmentRunReport) -> str:
    """Render an enrichment report as deterministic JSON."""
    return json.dumps(asdict(report), default=str, sort_keys=True, indent=2)


async def _top_skills(run_id: int, top_n: int) -> list[TopSkillRow]:
    async with async_session_factory() as session:
        rows = await session.execute(
            select(
                Skill.canonical,
                func.count(JobSkill.id).label("skill_rows"),
                func.count(func.distinct(JobSkill.job_id)).label("jobs"),
            )
            .join(JobSkill, JobSkill.skill_id == Skill.id)
            .where(JobSkill.run_id == run_id)
            .group_by(Skill.canonical)
            .order_by(func.count(JobSkill.id).desc(), Skill.canonical.asc())
            .limit(top_n)
        )
        return [
            TopSkillRow(canonical=row.canonical, skill_rows=int(row.skill_rows), jobs=int(row.jobs))
            for row in rows.all()
        ]


async def _source_breakdown(run_id: int) -> list[EnrichmentBreakdownRow]:
    async with async_session_factory() as session:
        primary_source = _primary_source_subquery()
        rows = await session.execute(
            select(
                primary_source.c.source,
                JobEnrichmentResult.status,
                JobEnrichmentResult.skill_count,
                JobEnrichmentResult.job_id,
            )
            .join(Job, Job.id == JobEnrichmentResult.job_id)
            .join(primary_source, primary_source.c.job_id == Job.id)
            .where(JobEnrichmentResult.run_id == run_id)
            .order_by(primary_source.c.source, JobEnrichmentResult.job_id)
        )
        return _aggregate_breakdown(
            [(row.source, row.status, int(row.skill_count), int(row.job_id)) for row in rows.all()]
        )


async def _country_breakdown(run_id: int) -> list[EnrichmentBreakdownRow]:
    async with async_session_factory() as session:
        rows = await session.execute(
            select(
                func.coalesce(Job.country, "(unknown)").label("country"),
                JobEnrichmentResult.status,
                JobEnrichmentResult.skill_count,
                JobEnrichmentResult.job_id,
            )
            .join(Job, Job.id == JobEnrichmentResult.job_id)
            .where(JobEnrichmentResult.run_id == run_id)
            .order_by("country", JobEnrichmentResult.job_id)
        )
        return _aggregate_breakdown(
            [(row.country, row.status, int(row.skill_count), int(row.job_id)) for row in rows.all()]
        )


async def _failure_stage_breakdown(run_id: int) -> list[FailureStageRow]:
    async with async_session_factory() as session:
        rows = await session.execute(
            select(
                EnrichmentFailure.stage,
                func.count(EnrichmentFailure.id).label("failures"),
                func.sum(EnrichmentFailure.attempts).label("attempts"),
            )
            .where(EnrichmentFailure.run_id == run_id)
            .group_by(EnrichmentFailure.stage)
            .order_by(func.count(EnrichmentFailure.id).desc(), EnrichmentFailure.stage.asc())
        )
        return [
            FailureStageRow(
                stage=row.stage,
                failures=int(row.failures),
                attempts=int(row.attempts or 0),
            )
            for row in rows.all()
        ]


def _primary_source_subquery() -> Any:
    return (
        select(JobSource.job_id, func.min(RawJob.source).label("source"))
        .join(RawJob, RawJob.id == JobSource.raw_job_id)
        .group_by(JobSource.job_id)
        .subquery("primary_job_source")
    )


def _aggregate_breakdown(
    rows: list[tuple[str, str, int, int]],
) -> list[EnrichmentBreakdownRow]:
    by_key: dict[str, dict[int, tuple[str, int]]] = {}
    for key, status, skill_count, job_id in rows:
        by_key.setdefault(key, {})[job_id] = (status, skill_count)

    result: list[EnrichmentBreakdownRow] = []
    for key, jobs in by_key.items():
        statuses = list(jobs.values())
        succeeded = sum(1 for status, _count in statuses if status == "success")
        failed = sum(1 for status, _count in statuses if status == "failed")
        with_skills = sum(
            1 for status, skill_count in statuses if status == "success" and skill_count > 0
        )
        zero_skill = sum(
            1 for status, skill_count in statuses if status == "success" and skill_count == 0
        )
        result.append(
            EnrichmentBreakdownRow(
                key=key,
                jobs_processed=len(jobs),
                jobs_succeeded=succeeded,
                jobs_failed=failed,
                jobs_with_skills=with_skills,
                zero_skill_jobs=zero_skill,
            )
        )
    return sorted(result, key=lambda item: (-item.jobs_processed, item.key))


def _append_top_skills(lines: list[str], rows: list[TopSkillRow]) -> None:
    lines.extend(["Top skills", "----------"])
    if not rows:
        lines.append("(none)")
    else:
        lines.append(f"{'skill':<32} {'rows':>8} {'jobs':>8}")
        for row in rows:
            lines.append(f"{row.canonical:<32} {row.skill_rows:>8,} {row.jobs:>8,}")
    lines.append("")


def _append_breakdown(lines: list[str], title: str, rows: list[EnrichmentBreakdownRow]) -> None:
    lines.extend([title, "-" * len(title)])
    if not rows:
        lines.append("(none)")
    else:
        lines.append(
            f"{'key':<16} {'processed':>10} {'success':>8} {'failed':>8} "
            f"{'with skills':>12} {'zero':>8}"
        )
        for row in rows:
            lines.append(
                f"{row.key:<16} {row.jobs_processed:>10,} {row.jobs_succeeded:>8,} "
                f"{row.jobs_failed:>8,} {row.jobs_with_skills:>12,} "
                f"{row.zero_skill_jobs:>8,}"
            )
    lines.append("")


def _append_failure_stages(lines: list[str], rows: list[FailureStageRow]) -> None:
    lines.extend(["Failure stages", "--------------"])
    if not rows:
        lines.append("(none)")
    else:
        lines.append(f"{'stage':<24} {'failures':>10} {'attempts':>10}")
        for row in rows:
            lines.append(f"{row.stage:<24} {row.failures:>10,} {row.attempts:>10,}")


async def _scalar_int(statement: Any) -> int:
    async with async_session_factory() as session:
        value = await session.scalar(statement)
        return int(value or 0)
