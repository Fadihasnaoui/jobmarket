"""Tests for the persisted deterministic matcher enrichment runner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest
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
from jobmarket.skills.enrichment import (
    EnrichmentConfigError,
    MatcherEnrichmentConfig,
    run_matcher_enrichment,
)
from jobmarket.skills.matcher import SkillMatch, find_skills
from jobmarket.skills.ontology import SkillsOntology, load_ontology


@dataclass
class EnrichmentFixture:
    source: str
    company_id: int
    job_ids: list[int] = field(default_factory=list)
    raw_job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    inserted_skill_names: list[str] = field(default_factory=list)


@asynccontextmanager
async def _fixture() -> AsyncIterator[EnrichmentFixture]:
    suffix = uuid4().hex
    fixture = EnrichmentFixture(source=f"pytest-enrich-{suffix}", company_id=0)
    async with async_session_factory() as session:
        company = Company(
            name=f"Pytest Enrichment {suffix}",
            name_norm=f"pytest-enrichment-{suffix}",
        )
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        fixture.inserted_skill_names = await _ensure_ontology_skills_for_test(session)
        await session.commit()

    try:
        yield fixture
    finally:
        async with async_session_factory() as session:
            if fixture.run_ids:
                await session.execute(
                    delete(EnrichmentRun).where(EnrichmentRun.id.in_(fixture.run_ids))
                )
            if fixture.job_ids:
                await session.execute(delete(JobSkill).where(JobSkill.job_id.in_(fixture.job_ids)))
                await session.execute(
                    delete(JobEnrichmentResult).where(
                        JobEnrichmentResult.job_id.in_(fixture.job_ids)
                    )
                )
                await session.execute(
                    delete(EnrichmentFailure).where(EnrichmentFailure.job_id.in_(fixture.job_ids))
                )
            if fixture.raw_job_ids:
                await session.execute(
                    delete(JobSource).where(JobSource.raw_job_id.in_(fixture.raw_job_ids))
                )
            if fixture.job_ids:
                await session.execute(delete(Job).where(Job.id.in_(fixture.job_ids)))
            if fixture.raw_job_ids:
                await session.execute(delete(RawJob).where(RawJob.id.in_(fixture.raw_job_ids)))
            await session.execute(delete(Company).where(Company.id == fixture.company_id))
            if fixture.inserted_skill_names:
                await session.execute(
                    delete(Skill).where(Skill.canonical.in_(fixture.inserted_skill_names))
                )
            await session.commit()


async def _ensure_ontology_skills_for_test(session: Any) -> list[str]:
    ontology = load_ontology()
    existing = set(await session.scalars(select(Skill.canonical)))
    missing = [entry for entry in ontology.entries if entry.canonical not in existing]
    session.add_all(
        [Skill(canonical=entry.canonical, category=entry.category) for entry in missing]
    )
    return [entry.canonical for entry in missing]


async def _add_job(
    fixture: EnrichmentFixture,
    *,
    title: str,
    description: str,
    country: str = "FR",
    source: str | None = None,
) -> int:
    actual_source = source or fixture.source
    suffix = uuid4().hex
    async with async_session_factory() as session:
        job = Job(
            job_hash=f"pytest-enrichment-{suffix}",
            company_id=fixture.company_id,
            title=title,
            title_norm=title.casefold(),
            description=description,
            country=country,
        )
        raw = RawJob(source=actual_source, source_id=f"pytest-enrichment-{suffix}", payload={})
        session.add_all([job, raw])
        await session.flush()
        session.add(JobSource(job_id=job.id, raw_job_id=raw.id))
        await session.commit()
        fixture.job_ids.append(job.id)
        fixture.raw_job_ids.append(raw.id)
        return job.id


async def _remember_run(fixture: EnrichmentFixture, run_id: int) -> None:
    fixture.run_ids.append(run_id)


async def _skill_canonical_for_job(run_id: int, job_id: int) -> list[str]:
    async with async_session_factory() as session:
        rows = await session.execute(
            select(Skill.canonical)
            .join(JobSkill, JobSkill.skill_id == Skill.id)
            .where(JobSkill.run_id == run_id, JobSkill.job_id == job_id)
            .order_by(Skill.canonical)
        )
        return list(rows.scalars().all())


async def _job_skill_count(run_id: int) -> int:
    async with async_session_factory() as session:
        count = await session.scalar(
            select(func.count()).select_from(JobSkill).where(JobSkill.run_id == run_id)
        )
        return int(count or 0)


async def test_runner_persists_title_description_duplicate_and_zero_skill_results() -> None:
    async with _fixture() as fixture:
        title_job = await _add_job(
            fixture,
            title="Senior Python Engineer",
            description="Build internal products.",
        )
        description_job = await _add_job(
            fixture,
            title="Data Platform Engineer",
            description="Own SQL models and AWS services.",
        )
        duplicate_job = await _add_job(
            fixture,
            title="Python Engineer",
            description="Python work happens here.",
        )
        zero_job = await _add_job(
            fixture,
            title="Customer Success Lead",
            description="Coordinate onboarding and stakeholder communication.",
        )

        summary = await run_matcher_enrichment(
            MatcherEnrichmentConfig(source=fixture.source, country="FR", batch_size=2)
        )
        await _remember_run(fixture, summary.run_id)

        assert summary.status == "success"
        assert summary.jobs_total == 4
        assert summary.jobs_processed == 4
        assert summary.jobs_succeeded == 4
        assert summary.jobs_failed == 0
        assert summary.jobs_with_skills == 3
        assert summary.zero_skill_jobs == 1

        assert await _skill_canonical_for_job(summary.run_id, title_job) == ["Python"]
        assert await _skill_canonical_for_job(summary.run_id, description_job) == ["AWS", "SQL"]
        assert await _skill_canonical_for_job(summary.run_id, duplicate_job) == ["Python"]
        assert await _skill_canonical_for_job(summary.run_id, zero_job) == []

        async with async_session_factory() as session:
            title_skill = await session.scalar(
                select(JobSkill).where(
                    JobSkill.run_id == summary.run_id,
                    JobSkill.job_id == title_job,
                )
            )
            assert title_skill is not None
            assert title_skill.evidence_field == "title"
            assert title_skill.evidence_text == "Python"
            assert title_skill.start_char == "Senior ".__len__()
            assert "Senior Python Engineer"[
                title_skill.start_char : title_skill.end_char
            ] == title_skill.evidence_text

            description_skill = await session.scalar(
                select(JobSkill)
                .join(Skill, Skill.id == JobSkill.skill_id)
                .where(
                    JobSkill.run_id == summary.run_id,
                    JobSkill.job_id == description_job,
                    Skill.canonical == "SQL",
                )
            )
            assert description_skill is not None
            assert description_skill.evidence_field == "description"
            assert description_skill.evidence_text == "SQL"
            assert "Own SQL models and AWS services."[
                description_skill.start_char : description_skill.end_char
            ] == description_skill.evidence_text

            zero_result = await session.scalar(
                select(JobEnrichmentResult).where(
                    JobEnrichmentResult.run_id == summary.run_id,
                    JobEnrichmentResult.job_id == zero_job,
                )
            )
            assert zero_result is not None
            assert zero_result.status == "success"
            assert zero_result.skill_count == 0


async def test_runner_filters_by_source_country_id_range_and_limit() -> None:
    async with _fixture() as fixture:
        excluded_country = await _add_job(
            fixture,
            title="Python Engineer",
            description="",
            country="US",
        )
        first = await _add_job(fixture, title="SQL Engineer", description="", country="FR")
        second = await _add_job(fixture, title="AWS Engineer", description="", country="FR")
        await _add_job(
            fixture,
            title="Python Engineer",
            description="",
            country="FR",
            source=f"{fixture.source}-other",
        )

        summary = await run_matcher_enrichment(
            MatcherEnrichmentConfig(
                source=fixture.source,
                country="FR",
                min_job_id=excluded_country + 1,
                max_job_id=second,
                limit=1,
                batch_size=1,
            )
        )
        await _remember_run(fixture, summary.run_id)

        assert summary.jobs_total == 1
        assert summary.jobs_processed == 1
        assert await _skill_canonical_for_job(summary.run_id, first) == ["SQL"]
        assert await _skill_canonical_for_job(summary.run_id, second) == []


async def test_runner_is_idempotent_and_force_creates_new_history() -> None:
    async with _fixture() as fixture:
        await _add_job(fixture, title="Python Engineer", description="", country="FR")
        config = MatcherEnrichmentConfig(source=fixture.source, country="FR")

        first = await run_matcher_enrichment(config)
        await _remember_run(fixture, first.run_id)
        second = await run_matcher_enrichment(config)
        await _remember_run(fixture, second.run_id)
        forced = await run_matcher_enrichment(config, force=True)
        await _remember_run(fixture, forced.run_id)

        assert first.run_id != second.run_id != forced.run_id
        assert first.jobs_total == 1
        assert first.jobs_processed == 1
        assert second.status == "success"
        assert second.jobs_total == 0
        assert second.jobs_processed == 0
        assert await _job_skill_count(second.run_id) == 0
        assert forced.jobs_total == 1
        assert forced.jobs_processed == 1
        assert await _job_skill_count(forced.run_id) == 1


async def test_runner_resumes_failed_jobs_and_reuses_same_run_id() -> None:
    async with _fixture() as fixture:
        await _add_job(fixture, title="SQL Engineer", description="", country="FR")
        failing_job = await _add_job(fixture, title="Python Engineer", description="", country="FR")
        config = MatcherEnrichmentConfig(source=fixture.source, country="FR", batch_size=1)

        def fail_python(text: str, ontology: SkillsOntology) -> list[SkillMatch]:
            if "Python" in text:
                raise RuntimeError("planned matcher failure")
            return find_skills(text, ontology)

        first = await run_matcher_enrichment(config, match_func=fail_python)
        await _remember_run(fixture, first.run_id)
        resumed = await run_matcher_enrichment(config, resume_run_id=first.run_id)

        assert resumed.run_id == first.run_id
        assert first.status == "partial"
        assert first.jobs_succeeded == 1
        assert first.jobs_failed == 1
        assert resumed.status == "success"
        assert resumed.jobs_succeeded == 2
        assert resumed.jobs_failed == 0
        assert await _skill_canonical_for_job(resumed.run_id, failing_job) == ["Python"]

        async with async_session_factory() as session:
            failure = await session.scalar(
                select(EnrichmentFailure).where(
                    EnrichmentFailure.run_id == first.run_id,
                    EnrichmentFailure.job_id == failing_job,
                )
            )
            assert failure is not None
            assert failure.attempts == 1


async def test_runner_updates_failure_attempts_on_repeated_resume_failure() -> None:
    async with _fixture() as fixture:
        await _add_job(fixture, title="Python Engineer", description="", country="FR")
        config = MatcherEnrichmentConfig(source=fixture.source, country="FR")

        def fail_all(_text: str, _ontology: SkillsOntology) -> list[SkillMatch]:
            raise RuntimeError("still broken")

        first = await run_matcher_enrichment(config, match_func=fail_all)
        await _remember_run(fixture, first.run_id)
        second = await run_matcher_enrichment(
            config,
            resume_run_id=first.run_id,
            match_func=fail_all,
        )

        assert first.status == "failed"
        assert second.status == "failed"
        assert second.run_id == first.run_id
        assert second.jobs_failed == 1

        async with async_session_factory() as session:
            failure = await session.scalar(
                select(EnrichmentFailure).where(EnrichmentFailure.run_id == first.run_id)
            )
            assert failure is not None
            assert failure.attempts == 2


async def test_resume_rejects_config_mismatch() -> None:
    async with _fixture() as fixture:
        await _add_job(fixture, title="Python Engineer", description="", country="FR")
        first = await run_matcher_enrichment(
            MatcherEnrichmentConfig(source=fixture.source, country="FR")
        )
        await _remember_run(fixture, first.run_id)

        with pytest.raises(EnrichmentConfigError, match="config does not match"):
            await run_matcher_enrichment(
                MatcherEnrichmentConfig(source=fixture.source, country="US"),
                resume_run_id=first.run_id,
            )


async def test_resume_and_force_are_mutually_exclusive() -> None:
    with pytest.raises(EnrichmentConfigError, match="cannot be used together"):
        await run_matcher_enrichment(
            MatcherEnrichmentConfig(source="does-not-matter"),
            resume_run_id=1,
            force=True,
        )
