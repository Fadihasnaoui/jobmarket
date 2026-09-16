"""Integration tests for the RAG chat answer path.

`search_jobs_by_embedding` itself (pgvector cosine search) is monkeypatched to return
fixed hits pointing at real fixture jobs — this tests the part that matters for
grounding (detail-fetch + context-building + the LLM call's constraints), not
pgvector's own kNN correctness, which isn't this project's code. The embedding
encoder is a fixed-vector fake (same convention as `test_api_cv.py::FakeEncoder`), and
the LLM client is a `MagicMock`, exactly like `test_cv_llm_extraction.py`. Zero real
network access.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

from sqlalchemy import delete, select

from jobmarket.chat.rag_answerer import _fetch_job_details, answer_rag_question
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.embeddings.repository import SemanticJobHit
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


def _mock_llm(content: str) -> MagicMock:
    client = MagicMock()
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


@dataclass
class ChatRagFixture:
    company_id: int = 0
    job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    skill_ids: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def _fixture() -> AsyncIterator[ChatRagFixture]:
    suffix = uuid4().hex
    fixture = ChatRagFixture()
    async with async_session_factory() as session:
        company = Company(name=f"Chat RAG Test {suffix}", name_norm=f"chat-rag-test-{suffix}")
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        existing = await session.scalar(select(Skill).where(Skill.canonical == "Python"))
        if existing is None:
            existing = Skill(canonical="Python", category="tool")
            session.add(existing)
            await session.flush()
        fixture.skill_ids["Python"] = existing.id
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


async def _add_run(session: object, fixture: ChatRagFixture, *, jobs_total: int) -> int:
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
    session: object, fixture: ChatRagFixture, title: str, *, with_skill: bool, run_id: int | None
) -> int:
    job = Job(
        job_hash=f"chat-rag-test-{uuid4().hex}",
        company_id=fixture.company_id,
        title=title,
        title_norm=title.casefold(),
        description="A real job description mentioning Python and machine learning.",
        location_raw="Berlin",
        country="de",
        contract_type="full_time",
        is_remote=True,
        url="https://example.test/job",
    )
    session.add(job)  # type: ignore[attr-defined]
    await session.flush()  # type: ignore[attr-defined]
    fixture.job_ids.append(job.id)
    if run_id is not None:
        if with_skill:
            session.add(  # type: ignore[attr-defined]
                JobSkill(
                    run_id=run_id,
                    job_id=job.id,
                    skill_id=fixture.skill_ids["Python"],
                    method="matcher",
                    matched_alias="Python",
                    evidence_text="Python",
                    evidence_field="description",
                    start_char=0,
                    end_char=6,
                )
            )
        session.add(  # type: ignore[attr-defined]
            JobEnrichmentResult(
                run_id=run_id, job_id=job.id, status="success", skill_count=1 if with_skill else 0
            )
        )
    return job.id


async def test_fetch_job_details_includes_job_outside_the_runs_coverage() -> None:
    """Regression test for the real bug found live: `search_jobs_by_embedding` finds
    jobs from the WHOLE embedded corpus, a broader population than any one
    enrichment run's coverage. Requiring run-membership to show job details silently
    dropped valid semantic hits -- this must not regress.
    """
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=1)
        in_run_id = await _add_job(session, fixture, "In-Run Job", with_skill=True, run_id=run_id)
        outside_run_id = await _add_job(
            session, fixture, "Outside-Run Job", with_skill=False, run_id=None
        )
        await session.commit()

        details = await _fetch_job_details(
            session, run_id, {in_run_id: 0.9, outside_run_id: 0.8}
        )

    by_id = {job.job_id: job for job in details}
    assert in_run_id in by_id
    assert outside_run_id in by_id  # must NOT be silently dropped
    assert by_id[in_run_id].skills == ["Python"]
    assert by_id[outside_run_id].skills == []  # no enrichment in this run, not an error


async def test_rag_answer_is_grounded_only_in_retrieved_jobs() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=1)
        job_id = await _add_job(
            session, fixture, "Python Backend Engineer", with_skill=True, run_id=run_id
        )
        await session.commit()

        async def fake_search(*args: object, **kwargs: object) -> list[SemanticJobHit]:
            return [
                SemanticJobHit(
                    job_id=job_id,
                    title="Python Backend Engineer",
                    company="Chat RAG Test Co",
                    country="de",
                    city="Berlin",
                    contract_type="full_time",
                    is_remote=True,
                    cosine_distance=0.1,
                    semantic_score=0.9,
                    semantic_rank=1,
                )
            ]

        import jobmarket.chat.rag_answerer as rag_module

        original = rag_module.search_jobs_by_embedding
        rag_module.search_jobs_by_embedding = fake_search  # type: ignore[assignment]
        try:
            client = _mock_llm("Grounded answer citing the Python Backend Engineer role.")
            result = await answer_rag_question(
                session,
                "tell me about backend roles",
                run_id=run_id,
                encoder=FakeEncoder(),
                client=client,
            )
        finally:
            rag_module.search_jobs_by_embedding = original

    assert "Python Backend Engineer" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].job_id == job_id

    # The LLM call's system prompt must forbid outside knowledge and the user
    # message must actually contain the retrieved job as context.
    call_kwargs = client.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    user_msg = call_kwargs["messages"][-1]["content"]
    assert "ONLY" in system_msg
    assert "Python Backend Engineer" in user_msg


async def test_no_semantic_hits_declines_honestly_without_calling_llm() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=0)
        await session.commit()

        async def empty_search(*args: object, **kwargs: object) -> list[SemanticJobHit]:
            return []

        import jobmarket.chat.rag_answerer as rag_module

        original = rag_module.search_jobs_by_embedding
        rag_module.search_jobs_by_embedding = empty_search  # type: ignore[assignment]
        client = MagicMock()
        client.chat.completions.create.side_effect = AssertionError("LLM should not be called")
        try:
            result = await answer_rag_question(
                session,
                "tell me about underwater basket weaving jobs",
                run_id=run_id,
                encoder=FakeEncoder(),
                client=client,
            )
        finally:
            rag_module.search_jobs_by_embedding = original

    assert result.sources == []
    assert "couldn't find" in result.answer.lower()
    client.chat.completions.create.assert_not_called()
