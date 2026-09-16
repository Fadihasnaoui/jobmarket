"""Integration tests for POST /chat — full router -> answer-path -> response shape.

The SQL and platform paths need no LLM/encoder at all (verified by never overriding
`get_chat_llm_client` for those tests — if the code tried to build a real client with
no API key configured, it would raise loudly rather than silently succeed). The
off-topic and RAG paths get a mocked LLM client via `app.dependency_overrides`, same
convention as `test_api_cv.py::FakeEncoder`. Zero real network access anywhere here.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from jobmarket.api.app import create_app
from jobmarket.api.dependencies import get_chat_llm_client, get_upload_encoder
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


def _response(payload: dict[str, str] | str) -> SimpleNamespace:
    content = json.dumps(payload) if isinstance(payload, dict) else payload
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _mock_llm(payload: dict[str, str] | str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _response(payload)
    return client


@dataclass
class ChatApiFixture:
    company_id: int = 0
    job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    skill_ids: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def _fixture() -> AsyncIterator[ChatApiFixture]:
    suffix = uuid4().hex
    fixture = ChatApiFixture()
    async with async_session_factory() as session:
        company = Company(name=f"Chat API Test {suffix}", name_norm=f"chat-api-test-{suffix}")
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        existing = await session.scalar(select(Skill).where(Skill.canonical == "Docker"))
        if existing is None:
            existing = Skill(canonical="Docker", category="tool")
            session.add(existing)
            await session.flush()
        fixture.skill_ids["Docker"] = existing.id
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


async def _add_run(session: object, fixture: ChatApiFixture, *, jobs_total: int) -> int:
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
    session: object, fixture: ChatApiFixture, title: str, skills: list[str], run_id: int
) -> int:
    job = Job(
        job_hash=f"chat-api-test-{uuid4().hex}",
        company_id=fixture.company_id,
        title=title,
        title_norm=title.casefold(),
        description="Synthetic job description for chat API tests.",
        location_raw="Paris",
        country="fr",
        contract_type="full_time",
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


async def test_chat_analytical_question_returns_real_db_number_no_llm_configured() -> None:
    """No LLM dependency override at all -- if the SQL path tried to reach an LLM
    with no API key configured, it would raise, not silently return a plausible
    number. This is the strongest available proof the number came from the query.
    """
    app = create_app()
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=2)
        await _add_job(session, fixture, "Backend Dev", ["Docker"], run_id)
        await _add_job(session, fixture, "Frontend Dev", [], run_id)
        await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"question": "what percent of jobs need Docker?", "run_id": run_id},
            )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["route"] == "sql"
    assert payload["data"]["skill"] == "Docker"
    assert "50.0%" in payload["answer"]


async def test_chat_platform_question_answered_without_llm() -> None:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={"question": "how does job matching work?"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["route"] == "platform"
    assert "score" in payload["answer"].casefold()


async def test_chat_off_topic_question_declined_not_answered_from_world_knowledge() -> None:
    app = create_app()
    # The mocked LLM would answer "Paris" if its content were ever used as the reply
    # -- proving the off-topic path uses the fixed refusal message, never LLM output.
    mock_client = _mock_llm({"route": "off_topic", "reason": "general knowledge"})
    app.dependency_overrides[get_chat_llm_client] = lambda: mock_client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json={"question": "what is the capital of France?"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["route"] == "off_topic"
    assert payload["answer"] == (
        "I can only help with questions about the job market and this platform's data."
    )
    assert "Paris" not in payload["answer"]


async def test_chat_rag_question_grounded_in_retrieved_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app()
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=1)
        job_id = await _add_job(session, fixture, "Docker Platform Engineer", ["Docker"], run_id)
        await session.commit()

        from jobmarket.embeddings.repository import SemanticJobHit

        async def fake_search(*args: object, **kwargs: object) -> list[SemanticJobHit]:
            return [
                SemanticJobHit(
                    job_id=job_id,
                    title="Docker Platform Engineer",
                    company="Chat API Test Co",
                    country="fr",
                    city="Paris",
                    contract_type="full_time",
                    is_remote=False,
                    cosine_distance=0.1,
                    semantic_score=0.9,
                    semantic_rank=1,
                )
            ]

        import jobmarket.chat.rag_answerer as rag_module

        monkeypatch.setattr(rag_module, "search_jobs_by_embedding", fake_search)

        # First call (router classification) returns a route decision; second call (RAG
        # grounded-answer generation) returns prose -- a realistic two-different-calls
        # sequence, not the same JSON reused as if it were a natural-language answer.
        client_mock = MagicMock()
        client_mock.chat.completions.create.side_effect = [
            _response({"route": "rag", "reason": "on-topic exploratory question"}),
            _response("The Docker Platform Engineer role at Chat API Test Co is based in Paris."),
        ]
        app.dependency_overrides[get_chat_llm_client] = lambda: client_mock
        app.dependency_overrides[get_upload_encoder] = lambda: FakeEncoder()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={
                    "question": "tell me about platform engineering roles",
                    "run_id": run_id,
                },
            )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["route"] == "rag"
    assert "Docker Platform Engineer" in payload["answer"]
    assert len(payload["sources"]) == 1
    assert "Docker Platform Engineer" in payload["sources"][0]["label"]

async def test_chat_selected_match_returns_the_exact_job_details() -> None:
    app = create_app()
    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture, jobs_total=1)
        job_id = await _add_job(session, fixture, "Backend Dev", ["Docker"], run_id)
        await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/chat",
                json={"question": "Tell me more about this job", "job_id": job_id},
            )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["route"] == "rag"
    assert "Backend Dev at Chat API Test" in payload["answer"]
    assert "Description:" in payload["answer"]
    assert payload["data"]["job_id"] == job_id
    assert payload["sources"][0]["label"].startswith("Backend Dev at Chat API Test")
