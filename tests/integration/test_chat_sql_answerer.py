"""Integration tests for the SQL/aggregate chat answer path — real DB, zero LLM.

`answer_sql_question` takes no LLM client parameter at all, by construction (see
`sql_answerer.py`'s module docstring) — there is no code path here that could call
one. These tests verify the returned numbers against independently known fixture
data, proving the answer traces to the real query, not a guess.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from uuid import uuid4

from sqlalchemy import delete, select

from jobmarket.chat.entities import ExtractedEntities
from jobmarket.chat.sql_answerer import SqlIntent, answer_sql_question, detect_sql_intent
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology


@dataclass
class ChatSqlFixture:
    company_id: int = 0
    job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    skill_ids: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def _fixture() -> AsyncIterator[ChatSqlFixture]:
    suffix = uuid4().hex
    fixture = ChatSqlFixture()
    async with async_session_factory() as session:
        company = Company(name=f"Chat SQL Test {suffix}", name_norm=f"chat-sql-test-{suffix}")
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        for canonical in ["Docker", "Python", "Kubernetes"]:
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


async def _add_run(session: object, fixture: ChatSqlFixture, *, jobs_total: int) -> int:
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
    fixture: ChatSqlFixture,
    title: str,
    skills: list[str],
    run_id: int,
    *,
    country: str = "fr",
    salary_min: float | None = None,
    salary_max: float | None = None,
    salary_currency: str | None = None,
) -> int:
    job = Job(
        job_hash=f"chat-sql-test-{uuid4().hex}",
        company_id=fixture.company_id,
        title=title,
        title_norm=title.casefold(),
        description="Synthetic job description for chat SQL-answerer tests.",
        location_raw="Paris",
        country=country,
        contract_type="full_time",
        is_remote=False,
        salary_min=Decimal(str(salary_min)) if salary_min is not None else None,
        salary_max=Decimal(str(salary_max)) if salary_max is not None else None,
        salary_currency=salary_currency,
        salary_period="year" if salary_min is not None else None,
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


def test_detect_sql_intent_covers_all_example_questions() -> None:
    empty = ExtractedEntities()
    with_skill = ExtractedEntities(skill="Docker")
    demand_intent = detect_sql_intent("what percent of jobs need Docker?", with_skill)
    assert demand_intent == SqlIntent.SKILL_DEMAND
    assert detect_sql_intent("how many jobs are in France?", empty) == SqlIntent.JOB_COUNT
    assert detect_sql_intent("top skills for Data Scientists?", empty) == SqlIntent.TOP_SKILLS
    assert detect_sql_intent("average salary of Data Engineers?", empty) == SqlIntent.AVG_SALARY


def test_detect_sql_intent_covers_french_example_questions() -> None:
    """Regression test: "quelles sont les compétences les plus demandées ?" was
    previously falling through to JOB_COUNT (no French top-skills wording at all in
    `detect_sql_intent`) instead of TOP_SKILLS -- the router correctly recognized the
    question as analytical, but the *intent* detector answered a different question
    than the one asked. Covers all four SQL intents in French.
    """
    empty = ExtractedEntities()
    python_skill = ExtractedEntities(skill="Python")
    assert detect_sql_intent("combien d'offres en France ?", empty) == SqlIntent.JOB_COUNT
    assert (
        detect_sql_intent("quelles sont les compétences les plus demandées ?", empty)
        == SqlIntent.TOP_SKILLS
    )
    assert (
        detect_sql_intent("quel est le salaire moyen d'un data scientist ?", empty)
        == SqlIntent.AVG_SALARY
    )
    assert (
        detect_sql_intent("quel pourcentage des offres demandent Python ?", python_skill)
        == SqlIntent.SKILL_DEMAND
    )


async def test_skill_demand_percentage_matches_real_fixture_counts() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=4)
        await _add_job(session, fixture, "Backend Dev", ["Docker", "Python"], run_id)
        await _add_job(session, fixture, "Platform Eng", ["Docker", "Kubernetes"], run_id)
        await _add_job(session, fixture, "Data Analyst", ["Python"], run_id)
        await _add_job(session, fixture, "Frontend Dev", [], run_id)
        await session.commit()

        ontology = load_ontology()
        result = await answer_sql_question(
            session, "what percent of jobs need Docker?", ontology, run_id=run_id
        )

    assert result.intent == SqlIntent.SKILL_DEMAND
    assert result.data["skill"] == "Docker"
    assert result.data["skill_job_count"] == 2
    assert result.data["total_job_count"] == 4
    assert result.data["pct"] == 50.0
    assert "50.0%" in result.answer
    assert "Docker" in result.answer


async def test_job_count_filters_by_country_matches_real_fixture() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=3)
        await _add_job(session, fixture, "Backend Dev", ["Python"], run_id, country="fr")
        await _add_job(session, fixture, "Platform Eng", ["Docker"], run_id, country="fr")
        await _add_job(session, fixture, "Data Analyst", ["Python"], run_id, country="de")
        await session.commit()

        ontology = load_ontology()
        result = await answer_sql_question(
            session, "how many jobs are in France?", ontology, run_id=run_id
        )

    assert result.intent == SqlIntent.JOB_COUNT
    assert result.data["job_count"] == 2
    assert "2" in result.answer


async def test_top_skills_ranks_by_real_demand_count() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=3)
        await _add_job(session, fixture, "Backend Dev", ["Python", "Docker"], run_id)
        await _add_job(session, fixture, "Platform Eng", ["Docker", "Kubernetes"], run_id)
        await _add_job(session, fixture, "Data Analyst", ["Python"], run_id)
        await session.commit()

        ontology = load_ontology()
        result = await answer_sql_question(
            session, "what are the top skills?", ontology, run_id=run_id
        )

    assert result.intent == SqlIntent.TOP_SKILLS
    top_skills = {item["skill"]: item["job_count"] for item in result.data["top_skills"]}
    assert top_skills["Python"] == 2
    assert top_skills["Docker"] == 2
    assert top_skills["Kubernetes"] == 1
    assert result.data["top_skills"][0]["skill"] in {"Python", "Docker"}  # tied for first


async def test_average_salary_converts_currency_and_excludes_placeholder_zeros() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=3)
        # 60,000 EUR and 80,000 EUR -> real average 70,000 EUR
        await _add_job(
            session, fixture, "Backend Dev", [], run_id,
            salary_min=60_000, salary_max=60_000, salary_currency="EUR",
        )
        await _add_job(
            session, fixture, "Platform Eng", [], run_id,
            salary_min=80_000, salary_max=80_000, salary_currency="EUR",
        )
        # RemoteOK-style 0/0 placeholder -- must be excluded, not averaged in as 0.
        await _add_job(
            session, fixture, "Zero Placeholder", [], run_id,
            salary_min=0, salary_max=0, salary_currency="EUR",
        )
        await session.commit()

        ontology = load_ontology()
        result = await answer_sql_question(
            session, "what is the average salary?", ontology, run_id=run_id
        )

    assert result.intent == SqlIntent.AVG_SALARY
    assert result.data["sample_size"] == 2
    assert result.data["avg_salary_eur"] == 70_000
    assert "70,000" in result.answer


async def test_average_salary_falls_back_to_country_currency_when_unset() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=1)
        # No explicit salary_currency -- must resolve via country (gb -> GBP) like the
        # validated salary-model pipeline does, not silently drop the row.
        await _add_job(
            session, fixture, "London Role", [], run_id,
            country="gb", salary_min=50_000, salary_max=50_000, salary_currency=None,
        )
        await session.commit()

        ontology = load_ontology()
        result = await answer_sql_question(
            session, "what is the average salary?", ontology, run_id=run_id
        )

    assert result.data["sample_size"] == 1
    # 50,000 GBP * 1.17 FX rate = 58,500 EUR
    assert result.data["avg_salary_eur"] == 58_500


async def test_no_matching_jobs_reports_honestly_instead_of_guessing() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=1)
        await _add_job(session, fixture, "Backend Dev", ["Python"], run_id, country="fr")
        await session.commit()

        ontology = load_ontology()
        result = await answer_sql_question(
            session, "what is the average salary in Germany?", ontology, run_id=run_id
        )

    assert result.data.get("sample_size", 0) == 0
    assert "don't have enough" in result.answer.lower()
