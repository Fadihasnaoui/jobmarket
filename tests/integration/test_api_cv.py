"""Integration tests for the CV API endpoints.

DB-backed (against the real dev Postgres, scoped/cleaned-up fixtures — same
convention as `tests/unit/test_cv_matching.py`; this project has no separate ephemeral
test database). All LLM and real-model-embedding calls are mocked — zero network
access and no sentence-transformers model load, matching the convention established in
`tests/unit/test_llm_extractor.py`/`test_cv_llm_extraction.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from jobmarket.api.app import create_app
from jobmarket.api.dependencies import get_cv_store, get_upload_encoder
from jobmarket.api.store import CvStore
from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.llm_extraction import CvExtractionResult
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology


class FakeEncoder:
    """Deterministic stand-in for the real sentence-transformers model."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 383 for _ in texts]


@dataclass
class ApiFixture:
    company_id: int = 0
    job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    skill_ids: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def _fixture() -> AsyncIterator[ApiFixture]:
    suffix = uuid4().hex
    fixture = ApiFixture()
    async with async_session_factory() as session:
        company = Company(name=f"API Test {suffix}", name_norm=f"api-test-{suffix}")
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        for canonical in ["Python", "AWS", "SQL", "Data Science", "Kubernetes"]:
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


async def _add_run(session: object, fixture: ApiFixture) -> int:
    versions = get_matcher_versions(load_ontology())
    run = EnrichmentRun(
        run_type="matcher",
        status="success",
        matcher_version=versions.matcher_version,
        ontology_version=versions.ontology_version,
        config={"pytest": True},
        jobs_total=1,
        jobs_processed=1,
        jobs_succeeded=1,
        jobs_with_skills=1,
        zero_skill_jobs=0,
    )
    session.add(run)  # type: ignore[attr-defined]
    await session.flush()  # type: ignore[attr-defined]
    fixture.run_ids.append(run.id)
    return run.id


async def _add_job(
    session: object,
    fixture: ApiFixture,
    title: str,
    skills: list[str],
    run_id: int,
    *,
    description: str = "Synthetic job description for API integration tests.",
) -> int:
    job = Job(
        job_hash=f"api-test-{uuid4().hex}",
        company_id=fixture.company_id,
        title=title,
        title_norm=title.casefold(),
        description=description,
        location_raw="Paris",
        country="fr",
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


def _fake_extract_deterministic(
    document: object,
    *,
    ontology: object = None,
    skill_ids: object = None,
    client: object = None,
):
    """Force the deterministic parser — no real Groq call in tests."""
    profile = extract_cv_profile(document, ontology=ontology, skill_ids=skill_ids)
    return CvExtractionResult(profile, [], "deterministic_fallback", "forced for test")


@pytest.fixture
def app_client() -> tuple[object, CvStore]:
    app = create_app()
    store = CvStore()
    app.dependency_overrides[get_cv_store] = lambda: store
    app.dependency_overrides[get_upload_encoder] = lambda: FakeEncoder()
    return app, store


def _strong_cv_text() -> str:
    return (
        "Profile\nSoftware engineer.\n\n"
        "Experience\nJan 2023 - Dec 2023 Python Developer - ExampleCo\n"
        "Built APIs with Python and deployed on AWS.\n\n"
        "Education\nBachelor in Computer Science, Example University, graduated 2022.\n\n"
        "Skills\nPython, AWS\n"
    )


async def test_upload_extract_and_match_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    app_client: tuple[object, CvStore],
) -> None:
    app, _store = app_client
    monkeypatch.setattr(
        "jobmarket.api.cv_routes.extract_cv_profile_with_fallback", _fake_extract_deterministic
    )

    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture)
        job_id = await _add_job(session, fixture, "Python Developer", ["Python", "AWS"], run_id)
        await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload_resp = await client.post(
                "/cv/upload",
                files={"file": ("cv.txt", _strong_cv_text().encode(), "text/plain")},
            )
            assert upload_resp.status_code == 200
            upload_payload = upload_resp.json()
            assert upload_payload["cv_id"]
            # extract_cv_profile_with_fallback was forced to the deterministic path,
            # whose experience-title matcher only recognizes a narrow, hardcoded set
            # (Sales Assistant, Insurance Intern, ...) — "Python Developer" isn't one
            # of them, so it genuinely finds zero experience entries here. That's the
            # real, honest behavior of the fallback parser, not a bug in the API.
            assert upload_payload["extraction_status"] == "insufficient_profile_evidence"
            assert upload_payload["extraction_method"] == "deterministic_fallback"
            assert upload_payload["extraction_degraded"] is True
            assert upload_payload["extraction_degraded_reason"] == "forced for test"
            assert "llm_extraction_degraded" in upload_payload["warnings"]
            assert "LLM-based extraction failed" in upload_payload["message"]
            assert "Python" in upload_payload["profile"]["canonical_skills"]

            cv_id = upload_payload["cv_id"]
            matches_resp = await client.get(
                f"/cv/{cv_id}/matches",
                params={"mode": "lexical", "run_id": run_id, "limit": 10},
            )
            assert matches_resp.status_code == 200
            matches_payload = matches_resp.json()
            assert matches_payload["out_of_scope"] is False
            # The earlier "never block on low confidence" fix must still hold through
            # this new API path: low_confidence=True, but real matches are returned,
            # not an empty list.
            assert matches_payload["low_confidence"] is True
            assert matches_payload["matches"]
            top = matches_payload["matches"][0]
            assert top["job_id"] == job_id
            assert "Python" in top["matched_skills"]
            assert top["final_score"] > 0
            assert "Recommended because" in top["explanation"]


async def test_matches_unknown_cv_id_returns_404(app_client: tuple[object, CvStore]) -> None:
    app, _store = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/cv/does-not-exist/matches")
    assert resp.status_code == 404


async def test_upload_rejects_unsupported_extension(app_client: tuple[object, CvStore]) -> None:
    app, _store = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/cv/upload",
            files={"file": ("cv.exe", b"not a cv", "application/octet-stream")},
        )
    assert resp.status_code == 422


async def test_upload_rejects_empty_file(app_client: tuple[object, CvStore]) -> None:
    app, _store = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/cv/upload", files={"file": ("cv.txt", b"", "text/plain")})
    assert resp.status_code == 422


async def test_scanned_pdf_like_empty_document_returns_soft_status(
    app_client: tuple[object, CvStore],
) -> None:
    """A .txt file with only whitespace mirrors an unparseable document without a PDF."""
    app, _store = app_client
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/cv/upload", files={"file": ("cv.txt", b"   \n  ", "text/plain")}
        )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["cv_id"] is None
    assert payload["document_status"] == "parse_error"
    assert payload["profile"] is None


async def test_out_of_scope_profile_short_circuits_matches_and_skill_gap(
    monkeypatch: pytest.MonkeyPatch,
    app_client: tuple[object, CvStore],
) -> None:
    app, _store = app_client
    monkeypatch.setattr(
        "jobmarket.api.cv_routes.extract_cv_profile_with_fallback", _fake_extract_deterministic
    )
    sales_cv_text = (
        "Profile\nSales assistant with strong customer relationships.\n\n"
        "Experience\nJan 2023 - Dec 2023 Sales Assistant - ExampleCo\n"
        "Vente de produits, relation client, service client, encaissement.\n"
    )

    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture)
        await _add_job(session, fixture, "Python Developer", ["Python"], run_id)
        await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload_resp = await client.post(
                "/cv/upload",
                files={"file": ("sales.txt", sales_cv_text.encode(), "text/plain")},
            )
            assert upload_resp.status_code == 200
            upload_payload = upload_resp.json()
            assert upload_payload["out_of_scope"] is True
            assert "outside the IT/tech scope" in upload_payload["message"]
            cv_id = upload_payload["cv_id"]

            matches_resp = await client.get(
                f"/cv/{cv_id}/matches", params={"run_id": run_id}
            )
            matches_payload = matches_resp.json()
            assert matches_payload["out_of_scope"] is True
            assert matches_payload["matches"] == []

            gap_resp = await client.get(f"/cv/{cv_id}/skill-gap", params={"run_id": run_id})
            gap_payload = gap_resp.json()
            assert gap_payload["out_of_scope"] is True
            assert gap_payload["gaps"] == []


async def test_skill_gap_ranks_missing_skills_by_demand(
    monkeypatch: pytest.MonkeyPatch,
    app_client: tuple[object, CvStore],
) -> None:
    app, _store = app_client
    monkeypatch.setattr(
        "jobmarket.api.cv_routes.extract_cv_profile_with_fallback", _fake_extract_deterministic
    )

    async with _fixture() as fixture, async_session_factory() as session:
        run_id = await _add_run(session, fixture)
        await _add_job(
            session, fixture, "ML Engineer", ["Python", "AWS", "Data Science"], run_id
        )
        await _add_job(
            session, fixture, "Data Engineer", ["Python", "Kubernetes", "SQL"], run_id
        )
        await session.commit()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            upload_resp = await client.post(
                "/cv/upload",
                files={"file": ("cv.txt", _strong_cv_text().encode(), "text/plain")},
            )
            cv_id = upload_resp.json()["cv_id"]

            gap_resp = await client.get(
                f"/cv/{cv_id}/skill-gap",
                params={"run_id": run_id, "mode": "lexical", "top": 10},
            )
            assert gap_resp.status_code == 200
            gap_payload = gap_resp.json()
            assert gap_payload["out_of_scope"] is False
            gap_skills = {item["canonical_skill"] for item in gap_payload["gaps"]}
            # CV only has Python/AWS; both jobs require other skills the CV lacks.
            assert gap_skills & {"Data Science", "Kubernetes", "SQL"}
            assert "AWS" not in gap_skills  # AWS is not missing, the CV already has it
