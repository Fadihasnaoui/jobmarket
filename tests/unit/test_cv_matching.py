"""Tests for deterministic CV-to-job ranking."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from jobmarket.cv.attributes import extract_candidate_attributes
from jobmarket.cv.matching import DEFAULT_RECOMMENDATION_RUN_ID, recommend_jobs_for_cv, score_values
from jobmarket.cv.profile import (
    CandidateAttributes,
    CvProfile,
    CvSkill,
    DocumentPage,
    EvidenceSpan,
    ExtractionVersions,
    ParsedDocument,
)
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.embeddings.repository import SemanticJobHit
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import load_ontology


@dataclass
class RankingFixture:
    company_id: int = 0
    job_ids: list[int] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)
    skill_ids: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def _fixture() -> AsyncIterator[RankingFixture]:
    suffix = uuid4().hex
    fixture = RankingFixture()
    async with async_session_factory() as session:
        company = Company(name=f"CV Match {suffix}", name_norm=f"cv-match-{suffix}")
        session.add(company)
        await session.flush()
        fixture.company_id = company.id
        for canonical in [
            "Python",
            "AWS",
            "Data Science",
            "Kubernetes",
            "Machine Learning",
            "Generative AI",
            "Retrieval-Augmented Generation",
            "Large Language Models",
            "Natural Language Processing",
            "Deep Learning",
        ]:
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


async def test_score_formula() -> None:
    coverage, overlap, final = score_values(
        matched_skill_count=2,
        job_skill_count=4,
        cv_skill_count=5,
    )

    assert coverage == 0.5
    assert overlap == 0.4
    assert final == pytest.approx(0.5425)


async def test_ranking_tie_breaking_zero_skill_exclusion_and_run_isolation() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        other_run = await _add_run(session, fixture, run_id=None)
        job_a = await _add_job(session, fixture, "A", ["Python", "AWS"], current_run)
        job_b = await _add_job(session, fixture, "B", ["Python"], current_run)
        job_c = await _add_job(session, fixture, "C", ["Python", "Kubernetes"], current_run)
        await _add_job(session, fixture, "Zero", [], current_run)
        await _add_job(session, fixture, "Other Run", ["AWS"], other_run)
        await session.commit()

        profile = _profile(["Python", "AWS"], fixture.skill_ids)
        recommendations = await recommend_jobs_for_cv(
            session,
            profile,
            run_id=current_run,
            top=10,
        )

    assert [item.job_id for item in recommendations] == [job_a, job_b, job_c]
    assert recommendations[0].final_score == pytest.approx(83.94)
    assert recommendations[1].final_score == pytest.approx(76.88)
    assert recommendations[2].final_score == pytest.approx(63.88)
    assert recommendations[0].score_components is not None
    assert recommendations[0].score_components.skill_score == pytest.approx(0.9487)
    assert recommendations[0].score_components.weighted_score_before_penalty == pytest.approx(
        0.8394
    )
    assert all(item.job_id != 0 for item in recommendations)
    assert "Other Run" not in {item.title for item in recommendations}
    assert DEFAULT_RECOMMENDATION_RUN_ID == 113


async def test_business_profile_suppresses_technical_jobs_without_skill_evidence() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        technical = await _add_job(
            session,
            fixture,
            "Data Science Developer",
            ["Python", "Data Science"],
            current_run,
            description="Data science and Python engineering role.",
        )
        await session.commit()

        profile = _profile([], fixture.skill_ids, text="Management customer service sales")
        profile = profile.model_copy(
            update={
                "candidate": CandidateAttributes(
                    career_level="junior",
                    preferred_domains=["Management", "Customer Service", "Sales"],
                )
            }
        )
        recommendations = await recommend_jobs_for_cv(
            session, profile, run_id=current_run, top=1
        )

    assert recommendations == []
    assert technical > 0


async def test_student_profile_penalizes_senior_freelance_roles() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        internship = await _add_job(
            session,
            fixture,
            "Data Science Internship",
            ["Python", "Data Science"],
            current_run,
            description="Stage data science remote avec Python, 6 months.",
            contract_type="Stage",
            is_remote=True,
        )
        senior = await _add_job(
            session,
            fixture,
            "Senior Freelance Data Scientist",
            ["Python", "Data Science"],
            current_run,
            description="Senior freelance mission requiring minimum 10 years experience.",
            contract_type="Freelance",
            is_remote=True,
        )
        await session.commit()

        profile = _profile(
            ["Python", "Data Science"],
            fixture.skill_ids,
            text="Student master data science internship, 6 months. Python Data Science.",
        )
        recommendations = await recommend_jobs_for_cv(
            session,
            profile,
            run_id=current_run,
            top=10,
        )

    assert [item.job_id for item in recommendations] == [internship, senior]
    assert recommendations[0].job_attributes.career_level == "internship"
    assert recommendations[0].penalties == []
    assert recommendations[1].job_attributes.career_level == "senior"
    assert {penalty.code for penalty in recommendations[1].penalties} == {
        "student_senior_mismatch",
        "large_experience_requirement",
        "freelance_low_experience",
        "experience_gap",
    }
    assert recommendations[1].final_score < recommendations[0].final_score
    assert "penalties=" in recommendations[1].explanation


async def test_ranking_is_deterministic_for_same_profile_and_run() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        await _add_job(session, fixture, "Python Role", ["Python"], current_run)
        await _add_job(session, fixture, "AWS Role", ["AWS"], current_run)
        await session.commit()

        profile = _profile(["Python", "AWS"], fixture.skill_ids)
        first = await recommend_jobs_for_cv(session, profile, run_id=current_run, top=10)
        second = await recommend_jobs_for_cv(session, profile, run_id=current_run, top=10)

    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]


async def test_student_rich_internship_beats_generic_and_senior_roles() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        rich = await _add_job(
            session,
            fixture,
            "Data Science Internship",
            ["Python", "AWS", "Data Science", "Machine Learning", "Kubernetes"],
            current_run,
            description="Stage data science with Python, AWS, Kubernetes and machine learning.",
            contract_type="Stage",
        )
        generic = await _add_job(
            session,
            fixture,
            "Generic Data Science Internship",
            ["Data Science", "Machine Learning"],
            current_run,
            description="Stage data science and machine learning.",
            contract_type="Stage",
        )
        senior = await _add_job(
            session,
            fixture,
            "Senior Data Science Lead",
            [
                "Python",
                "AWS",
                "Data Science",
                "Machine Learning",
                "Kubernetes",
                "Deep Learning",
            ],
            current_run,
            description="Senior lead role requiring minimum 10 years experience.",
            contract_type="CDI",
        )
        await session.commit()

        profile = _profile(
            [
                "Python",
                "AWS",
                "Data Science",
                "Machine Learning",
                "Kubernetes",
                "Deep Learning",
            ],
            fixture.skill_ids,
            text=(
                "Student in Paris. Stage 6 months. Python AWS Data Science "
                "Machine Learning Kubernetes Deep Learning."
            ),
        )
        recommendations = await recommend_jobs_for_cv(session, profile, run_id=current_run, top=3)

    assert [item.job_id for item in recommendations] == [rich, generic, senior]
    assert recommendations[0].score_components is not None
    assert recommendations[0].score_components.skill_profile_confidence > (
        recommendations[1].score_components.skill_profile_confidence
        if recommendations[1].score_components
        else 0
    )
    assert "student_senior_mismatch" in {penalty.code for penalty in recommendations[2].penalties}


async def test_genai_cv_prefers_specific_genai_internship() -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        genai = await _add_job(
            session,
            fixture,
            "GenAI Internship",
            [
                "Generative AI",
                "Retrieval-Augmented Generation",
                "Large Language Models",
                "Natural Language Processing",
            ],
            current_run,
            description="Stage GenAI with RAG, LLM and NLP prototypes.",
            contract_type="Stage",
        )
        generic = await _add_job(
            session,
            fixture,
            "Data Science Internship",
            ["Data Science", "Machine Learning"],
            current_run,
            description="Stage generic data science and machine learning.",
            contract_type="Stage",
        )
        await session.commit()

        profile = _profile(
            [
                "Generative AI",
                "Retrieval-Augmented Generation",
                "Large Language Models",
                "Natural Language Processing",
            ],
            fixture.skill_ids,
            text=(
                "Student GenAI internship 6 months. Generative AI "
                "Retrieval-Augmented Generation Large Language Models "
                "Natural Language Processing."
            ),
        )
        recommendations = await recommend_jobs_for_cv(session, profile, run_id=current_run, top=2)

    assert [item.job_id for item in recommendations] == [genai, generic]
    assert len(recommendations[0].specialized_matched_skills) == 4
    assert recommendations[0].score_components is not None
    assert recommendations[0].score_components.domain_score > (
        recommendations[1].score_components.domain_score
    )


async def test_hybrid_matching_combines_lexical_and_semantic_candidates(monkeypatch) -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        lexical = await _add_job(session, fixture, "Python Role", ["Python"], current_run)
        semantic = await _add_job(
            session, fixture, "Deep Learning Role", ["Deep Learning"], current_run
        )
        await session.commit()

        async def fake_search(session_arg, query_embedding, *, limit, filters):
            assert query_embedding == [1.0] + [0.0] * 383
            return [
                SemanticJobHit(
                    job_id=semantic,
                    title="Deep Learning Role",
                    company="Example",
                    country="FR",
                    city=None,
                    contract_type=None,
                    is_remote=None,
                    cosine_distance=0.1,
                    semantic_score=0.9,
                    semantic_rank=1,
                )
            ]

        monkeypatch.setattr("jobmarket.cv.matching.search_jobs_by_embedding", fake_search)
        monkeypatch.setattr(
            "jobmarket.cv.matching.embed_cv_profile",
            lambda profile, encoder=None: [1.0] + [0.0] * 383,
        )
        profile = _profile(["Python"], fixture.skill_ids)
        recommendations = await recommend_jobs_for_cv(
            session,
            profile,
            run_id=current_run,
            top=10,
            mode="hybrid",
            alpha=0.6,
            lexical_pool_size=1,
            semantic_pool_size=1,
        )

    assert {item.job_id for item in recommendations} == {lexical, semantic}
    semantic_item = next(item for item in recommendations if item.job_id == semantic)
    assert semantic_item.retrieval_source == "semantic"
    assert semantic_item.semantic_score == pytest.approx(0.9)
    assert semantic_item.semantic_rank == 1
    assert semantic_item.score_components is not None
    assert semantic_item.score_components.alpha == 0.6
    assert semantic_item.score_components.hybrid_score_before_penalty is not None


async def test_hybrid_never_zeros_out_lexical_only_candidates(monkeypatch) -> None:
    """A job outside the semantic pool keeps its lexical score, never blends with 0.0.

    Regression test for a real bug: `semantic_score` was collapsed to 0.0 for any job
    not returned by the semantic kNN search, then blended as if 0.0 were a genuine
    "no similarity at all" measurement — tanking otherwise-strong lexical matches by
    (1-alpha) of their weight. Missing evidence must fall back to the lexical score,
    not be treated as a real zero.
    """
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        lexical_only = await _add_job(session, fixture, "Python Role", ["Python"], current_run)
        await session.commit()

        async def empty_semantic_search(session_arg, query_embedding, *, limit, filters):
            return []

        monkeypatch.setattr(
            "jobmarket.cv.matching.search_jobs_by_embedding", empty_semantic_search
        )
        monkeypatch.setattr(
            "jobmarket.cv.matching.embed_cv_profile",
            lambda profile, encoder=None: [1.0] + [0.0] * 383,
        )
        profile = _profile(["Python"], fixture.skill_ids)

        lexical_recs = await recommend_jobs_for_cv(
            session, profile, run_id=current_run, top=10, mode="lexical"
        )
        hybrid_recs = await recommend_jobs_for_cv(
            session, profile, run_id=current_run, top=10, mode="hybrid", alpha=0.6
        )

    lexical_item = next(item for item in lexical_recs if item.job_id == lexical_only)
    hybrid_item = next(item for item in hybrid_recs if item.job_id == lexical_only)

    assert hybrid_item.retrieval_source == "lexical"
    assert hybrid_item.semantic_score is None
    assert hybrid_item.score_components is not None
    assert hybrid_item.score_components.semantic_score is None
    assert hybrid_item.final_score == lexical_item.final_score
    assert hybrid_item.score_components.hybrid_score_before_penalty == pytest.approx(
        hybrid_item.score_components.lexical_score
    )


async def test_hybrid_falls_back_to_lexical_when_embedding_model_is_unavailable(
    monkeypatch,
) -> None:
    async with _fixture() as fixture, async_session_factory() as session:
        current_run = await _add_run(session, fixture, run_id=None)
        lexical = await _add_job(session, fixture, "Python Role", ["Python"], current_run)
        await session.commit()

        def unavailable_encoder(*args, **kwargs):
            raise RuntimeError("local embedding model unavailable")

        monkeypatch.setattr("jobmarket.cv.matching.embed_cv_profile", unavailable_encoder)
        profile = _profile(["Python"], fixture.skill_ids)
        recommendations = await recommend_jobs_for_cv(
            session, profile, run_id=current_run, top=3, mode="hybrid"
        )

    assert [item.job_id for item in recommendations] == [lexical]
    assert recommendations[0].retrieval_source == "lexical"


def test_invalid_hybrid_alpha_is_rejected() -> None:
    async def _run() -> None:
        async with async_session_factory() as session:
            await recommend_jobs_for_cv(
                session, _profile([], {}), mode="hybrid", alpha=1.5
            )

    with pytest.raises(ValueError, match="alpha"):
        import asyncio

        asyncio.run(_run())


async def _add_run(session, fixture: RankingFixture, *, run_id: int | None) -> int:
    versions = get_matcher_versions(load_ontology())
    run = EnrichmentRun(
        id=run_id,
        run_type="matcher",
        status="success",
        matcher_version=versions.matcher_version,
        ontology_version=versions.ontology_version,
        config={"pytest": True},
        jobs_total=0,
        jobs_processed=0,
        jobs_succeeded=0,
        jobs_failed=0,
        jobs_with_skills=0,
        zero_skill_jobs=0,
    )
    session.add(run)
    await session.flush()
    fixture.run_ids.append(run.id)
    return run.id


async def _add_job(
    session,
    fixture: RankingFixture,
    title: str,
    skills: list[str],
    run_id: int,
    *,
    description: str = "Synthetic job",
    contract_type: str | None = None,
    is_remote: bool | None = None,
) -> int:
    job = Job(
        job_hash=f"cv-match-{uuid4().hex}",
        company_id=fixture.company_id,
        title=title,
        title_norm=title.casefold(),
        description=description,
        location_raw="Remote",
        country="FR",
        contract_type=contract_type,
        is_remote=is_remote,
    )
    session.add(job)
    await session.flush()
    fixture.job_ids.append(job.id)
    for canonical in skills:
        session.add(
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
    session.add(
        JobEnrichmentResult(
            run_id=run_id,
            job_id=job.id,
            status="success",
            skill_count=len(skills),
        )
    )
    return job.id


def _profile(
    canonicals: list[str],
    skill_ids: dict[str, int],
    *,
    text: str | None = None,
) -> CvProfile:
    text = text or " ".join(canonicals)
    document = ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text=text,
        page_count=None,
        pages=[DocumentPage(page=None, text=text, start_offset=0, end_offset=len(text))],
        warnings=[],
        file_hash="a" * 64,
        file_size=len(text.encode()),
    )
    versions = get_matcher_versions(load_ontology())
    offset = 0
    skills: list[CvSkill] = []
    for canonical in canonicals:
        start = text.index(canonical, offset)
        end = start + len(canonical)
        offset = end
        skills.append(
            CvSkill(
                canonical_skill=canonical,
                ontology_skill_id=skill_ids[canonical],
                matched_alias=canonical,
                extraction_method="deterministic",
                evidence=EvidenceSpan(
                    evidence_text=canonical,
                    page=None,
                    document_start=start,
                    document_end=end,
                    page_start=None,
                    page_end=None,
                ),
                matcher_version=versions.matcher_version,
                ontology_version=versions.ontology_version,
                validation_status="deterministic",
                confidence=1.0,
            )
        )
    ontology = load_ontology()
    candidate = extract_candidate_attributes(text, set(canonicals), ontology)
    return CvProfile(
        document=document,
        versions=ExtractionVersions(
            parser_version=document.parser_version,
            matcher_version=versions.matcher_version,
            ontology_version=versions.ontology_version,
        ),
        skills=skills,
        candidate=candidate,
    )
