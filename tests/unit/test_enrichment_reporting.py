"""Tests for persisted enrichment run reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import delete, func, select

from jobmarket.db.models import (
    Company,
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
from jobmarket.reporting.enrichment import (
    collect_enrichment_report,
    enrichment_report_to_json,
    format_enrichment_report,
)


async def test_enrichment_report_metrics_breakdowns_ordering_and_json_are_read_only() -> None:
    suffix = uuid4().hex[:8]
    job_ids: list[int] = []
    raw_ids: list[int] = []
    run_id: int | None = None
    company_id: int | None = None

    try:
        async with async_session_factory() as session:
            python_id = await session.scalar(select(Skill.id).where(Skill.canonical == "Python"))
            sql_id = await session.scalar(select(Skill.id).where(Skill.canonical == "SQL"))
            assert python_id is not None
            assert sql_id is not None

            company = Company(name=f"Report {suffix}", name_norm=f"report-{suffix}")
            session.add(company)
            await session.flush()
            company_id = company.id

            specs = [
                ("adzuna", "FR", "Python SQL Engineer", "Python and SQL."),
                ("remoteok", "US", "Operations Lead", "Coordinate launches."),
                ("adzuna", "FR", "Broken Matcher", "This job failed."),
                ("jooble", None, "Python Analyst", "Python reporting."),
            ]
            for index, (source, country, title, description) in enumerate(specs, start=1):
                job = Job(
                    job_hash=f"report-{suffix}-{index}",
                    company_id=company.id,
                    title=title,
                    title_norm=title.casefold(),
                    description=description,
                    country=country,
                )
                raw = RawJob(source=source, source_id=f"report-{suffix}-{index}", payload={})
                session.add_all([job, raw])
                await session.flush()
                session.add(JobSource(job_id=job.id, raw_job_id=raw.id))
                job_ids.append(job.id)
                raw_ids.append(raw.id)
                if index == 1:
                    duplicate_raw = RawJob(
                        source="remoteok",
                        source_id=f"report-{suffix}-{index}-duplicate",
                        payload={},
                    )
                    session.add(duplicate_raw)
                    await session.flush()
                    session.add(JobSource(job_id=job.id, raw_job_id=duplicate_raw.id))
                    raw_ids.append(duplicate_raw.id)

            run = EnrichmentRun(
                run_type="matcher",
                status="partial",
                matcher_version="matcher-test",
                ontology_version="ontology-test",
                config={"limit": 4, "batch_size": 2},
                jobs_total=4,
                jobs_processed=4,
                jobs_succeeded=3,
                jobs_failed=1,
                jobs_with_skills=2,
                zero_skill_jobs=1,
                started_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
            session.add(run)
            await session.flush()
            run_id = run.id

            session.add_all(
                [
                    JobEnrichmentResult(
                        run_id=run.id, job_id=job_ids[0], status="success", skill_count=2
                    ),
                    JobEnrichmentResult(
                        run_id=run.id, job_id=job_ids[1], status="success", skill_count=0
                    ),
                    JobEnrichmentResult(
                        run_id=run.id, job_id=job_ids[2], status="failed", skill_count=0
                    ),
                    JobEnrichmentResult(
                        run_id=run.id, job_id=job_ids[3], status="success", skill_count=1
                    ),
                    JobSkill(
                        run_id=run.id,
                        job_id=job_ids[0],
                        skill_id=python_id,
                        method="matcher",
                        evidence_field="title",
                        evidence_text="Python",
                        start_char=0,
                        end_char=6,
                    ),
                    JobSkill(
                        run_id=run.id,
                        job_id=job_ids[0],
                        skill_id=sql_id,
                        method="matcher",
                        evidence_field="title",
                        evidence_text="SQL",
                        start_char=7,
                        end_char=10,
                    ),
                    JobSkill(
                        run_id=run.id,
                        job_id=job_ids[3],
                        skill_id=python_id,
                        method="matcher",
                        evidence_field="title",
                        evidence_text="Python",
                        start_char=0,
                        end_char=6,
                    ),
                    EnrichmentFailure(
                        run_id=run.id,
                        job_id=job_ids[2],
                        stage="matcher",
                        attempts=2,
                        error="planned failure",
                    ),
                ]
            )
            await session.commit()

        assert run_id is not None
        before = await _table_counts(run_id)
        report = await collect_enrichment_report(run_id, top_n=10)
        after = await _table_counts(run_id)

        assert before == after
        assert report.status == "partial"
        assert report.persisted_skill_rows == 3
        assert report.mean_skills_per_successful_job == 1.0
        assert report.median_skills_per_successful_job == 1.0
        assert report.mean_skills_among_jobs_with_skills == 1.5
        assert report.zero_skill_rate == 1 / 3
        assert report.failure_rate == 0.25
        assert [(row.canonical, row.skill_rows) for row in report.top_skills] == [
            ("Python", 2),
            ("SQL", 1),
        ]
        assert [(row.key, row.jobs_processed) for row in report.source_breakdown] == [
            ("adzuna", 2),
            ("jooble", 1),
            ("remoteok", 1),
        ]
        assert [(row.key, row.jobs_processed) for row in report.country_breakdown] == [
            ("FR", 2),
            ("(unknown)", 1),
            ("US", 1),
        ]
        assert report.failure_stage_breakdown[0].stage == "matcher"
        assert report.failure_stage_breakdown[0].failures == 1
        assert report.failure_stage_breakdown[0].attempts == 2
        assert "Enrichment run report" in format_enrichment_report(report)
        assert '"run_id"' in enrichment_report_to_json(report)
    finally:
        async with async_session_factory() as session:
            if run_id is not None:
                await session.execute(delete(EnrichmentRun).where(EnrichmentRun.id == run_id))
            if raw_ids:
                await session.execute(delete(JobSource).where(JobSource.raw_job_id.in_(raw_ids)))
            if job_ids:
                await session.execute(delete(Job).where(Job.id.in_(job_ids)))
            if raw_ids:
                await session.execute(delete(RawJob).where(RawJob.id.in_(raw_ids)))
            if company_id is not None:
                await session.execute(delete(Company).where(Company.id == company_id))
            await session.commit()


async def _table_counts(run_id: int) -> dict[str, int]:
    async with async_session_factory() as session:
        return {
            "runs": int(await session.scalar(select(func.count()).select_from(EnrichmentRun)) or 0),
            "skills": int(
                await session.scalar(
                    select(func.count()).select_from(JobSkill).where(JobSkill.run_id == run_id)
                )
                or 0
            ),
            "results": int(
                await session.scalar(
                    select(func.count())
                    .select_from(JobEnrichmentResult)
                    .where(JobEnrichmentResult.run_id == run_id)
                )
                or 0
            ),
            "failures": int(
                await session.scalar(
                    select(func.count())
                    .select_from(EnrichmentFailure)
                    .where(EnrichmentFailure.run_id == run_id)
                )
                or 0
            ),
        }
