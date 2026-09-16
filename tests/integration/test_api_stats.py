"""Integration tests for the /stats endpoints — DB-backed, no external calls at all."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from jobmarket.api.app import create_app
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology


@dataclass
class StatsFixture:
    company_id: int = 0
    job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    skill_ids: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def _fixture() -> AsyncIterator[StatsFixture]:
    suffix = uuid4().hex
    fixture = StatsFixture()
    async with async_session_factory() as session:
        company = Company(name=f"Stats Test {suffix}", name_norm=f"stats-test-{suffix}")
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        for canonical in ["Python", "AWS", "Kubernetes"]:
            existing = await session.scalar(select(Skill).where(Skill.canonical == canonical))
            if existing is None:
                existing = Skill(canonical=canonical, category="tool")
                session.add(existing)
                await session.flush()
            fixture.skill_ids[canonical] = existing.id
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
                await session.execute(delete(Job).where(Job.id.in_(fixture.job_ids)))
            await session.execute(delete(Company).where(Company.id == fixture.company_id))
            await session.commit()


async def _add_run(session: object, fixture: StatsFixture, *, jobs_total: int) -> int:
    versions = get_matcher_versions(load_ontology())
    run = EnrichmentRun(
        run_type="matcher",
        status="success",
        matcher_version=versions.matcher_version,
        ontology_version=versions.ontology_version,
        config={"pytest": True},
        jobs_total=jobs_total,
        jobs_processed=jobs_total,
        jobs_succeeded=jobs_total,
        jobs_with_skills=jobs_total,
        zero_skill_jobs=0,
    )
    session.add(run)  # type: ignore[attr-defined]
    await session.flush()  # type: ignore[attr-defined]
    fixture.run_ids.append(run.id)
    return run.id


async def _add_job(
    session: object,
    fixture: StatsFixture,
    title: str,
    skills: list[str],
    run_id: int,
    *,
    country: str = "fr",
) -> int:
    job = Job(
        job_hash=f"stats-test-{uuid4().hex}",
        company_id=fixture.company_id,
        title=title,
        title_norm=title.casefold(),
        description="Synthetic job description for stats API tests.",
        location_raw="Paris",
        country=country,
        contract_type="cdi",
        is_remote=False,
    )
    session.add(job)  # type: ignore[attr-defined]
    await session.flush()  # type: ignore[attr-defined]
    fixture.job_ids.append(job.id)
    for canonical in skills:
        session.add(  # type: ignore[attr-defined]
            JobSkill(
                run_id=run_id,
                job_id=job.id,
                skill_id=fixture.skill_ids[canonical],
                method="matcher",
                matched_alias=canonical,
                evidence_text=canonical,
                evidence_field="description",
                start_char=0,
                end_char=len(canonical),
            )
        )
    session.add(  # type: ignore[attr-defined]
        JobEnrichmentResult(run_id=run_id, job_id=job.id, status="success", skill_count=len(skills))
    )
    return job.id


@pytest.fixture
def app_client() -> object:
    return create_app()


async def test_stats_overview_reports_reference_run_coverage(app_client: object) -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=2)
        await _add_job(session, fixture, "Python Dev", ["Python"], run_id)
        await _add_job(session, fixture, "Data Engineer", ["Python", "AWS"], run_id)
        await session.commit()

        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/stats/overview", params={"run_id": run_id})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["reference_run_id"] == run_id
    assert payload["jobs_in_reference_run"] == 2
    assert payload["jobs_with_skills"] == 2
    assert payload["skill_coverage_pct"] == 100.0
    assert payload["ontology_skill_count"] > 0
    assert payload["distinct_skills_matched"] >= 2
    assert payload["jobs"] > 0  # real corpus totals, non-zero
    assert isinstance(payload["per_source"], list)


async def test_stats_overview_unknown_run_returns_404(app_client: object) -> None:
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/stats/overview", params={"run_id": 999_999_999})
    assert resp.status_code == 404


async def test_stats_skills_ranks_by_demand_and_filters_by_country(app_client: object) -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=3)
        await _add_job(session, fixture, "Python Dev", ["Python"], run_id, country="fr")
        await _add_job(session, fixture, "Data Engineer", ["Python", "AWS"], run_id, country="fr")
        await _add_job(session, fixture, "DevOps", ["Kubernetes"], run_id, country="de")
        await session.commit()

        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            all_resp = await client.get(
                "/stats/skills", params={"run_id": run_id, "limit": 50}
            )
            fr_resp = await client.get(
                "/stats/skills", params={"run_id": run_id, "country": "fr", "limit": 50}
            )

    assert all_resp.status_code == 200
    all_payload = all_resp.json()
    by_skill = {item["canonical_skill"]: item["job_count"] for item in all_payload["skills"]}
    assert by_skill["Python"] == 2
    assert by_skill["AWS"] == 1
    assert by_skill["Kubernetes"] == 1
    # Ranked by demand: Python (2 jobs) must outrank AWS/Kubernetes (1 job each).
    assert all_payload["skills"][0]["canonical_skill"] == "Python"

    fr_payload = fr_resp.json()
    fr_skills = {item["canonical_skill"] for item in fr_payload["skills"]}
    assert "Kubernetes" not in fr_skills  # only the German job requires it
    assert fr_payload["total_jobs_considered"] == 2


async def test_stats_skills_rejects_unknown_seniority(app_client: object) -> None:
    transport = ASGITransport(app=app_client)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/stats/skills", params={"seniority": "not-a-real-level"}
        )
    assert resp.status_code == 422
