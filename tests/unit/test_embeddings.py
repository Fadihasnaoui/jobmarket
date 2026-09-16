"""Tests for semantic embedding text and vector helpers."""

from __future__ import annotations

import math

import pytest

from jobmarket.cv.profile import (
    CandidateAttributes,
    CvProfile,
    CvSkill,
    DocumentPage,
    EducationDetail,
    EvidenceSpan,
    ExperienceEntry,
    ExtractionVersions,
    ParsedDocument,
)
from jobmarket.embeddings.config import get_embedding_settings
from jobmarket.embeddings.cv import embed_cv_profile
from jobmarket.embeddings.encoder import prepare_vector, validate_vector, vector_to_pg
from jobmarket.embeddings.text import build_cv_embedding_text, build_job_embedding_text


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        assert all("SECRET" not in text for text in texts)
        return [[1.0] + [0.0] * 383 for _ in texts]


def test_build_job_embedding_text_is_structured_and_deterministic() -> None:
    job = {
        "title": "Senior Data Scientist",
        "description": "<p>Build NLP models and PyTorch pipelines.</p>" * 100,
        "contract_type": "CDI",
        "is_remote": True,
        "country": "fr",
        "city": "Paris",
    }
    enrichment = {"skills": ["PyTorch", "Natural Language Processing", "PyTorch"]}

    first = build_job_embedding_text(job, enrichment)
    second = build_job_embedding_text(job, enrichment)

    assert first == second
    assert "Title: Senior Data Scientist" in first
    assert "Skills: Natural Language Processing, PyTorch" in first
    assert "Domain:" in first
    assert "Data Science" in first
    assert "NLP" in first
    assert "Contract: CDI" in first
    assert "Work mode: remote" in first
    assert "<p>" not in first
    assert len(first.split("Description: ", 1)[1]) <= 1500


def test_build_job_embedding_text_omits_missing_fields() -> None:
    text = build_job_embedding_text(
        {"title": "Python Engineer", "description": "Python APIs", "is_remote": None},
        {"skills": []},
    )

    assert "Title: Python Engineer" in text
    assert "Skills:" not in text
    assert "Work mode:" not in text


def test_build_cv_embedding_text_includes_grounded_evidence_not_full_document() -> None:
    """Semantic similarity must see the CV's real prose, not just canonical tags.

    `education_detail`/`experience_entries` are the fields the extractors actually
    populate; their evidence quotes are genuine excerpts from the CV. The rest of
    the raw document text (unrelated to any extracted fact) must still not appear.
    """
    profile = _profile()

    text = build_cv_embedding_text(profile)

    assert "SECRET RAW CV" not in text
    assert "Career level: junior" in text
    assert "Skills: Python" in text
    assert "Domain: Data Science" in text
    assert "Programming languages: Python" in text
    assert "Databases: PostgreSQL" in text
    assert "Example University" in text
    assert "MSc AI at Example University" in text
    assert "Research Assistant" in text
    assert "Built an NLP project for internal tooling" in text


def test_build_cv_embedding_text_ignores_legacy_unpopulated_fields() -> None:
    """`education`/`roles`/`experience_statements` are dead fields no extractor sets.

    Regression guard: even if something does populate them, the embedding text must
    keep reading from `education_detail`/`experience_entries` instead, so a stray
    legacy value can't silently leak into semantic search input unexamined.
    """
    profile = _profile().model_copy(
        update={
            "education_detail": None,
            "experience_entries": [],
            "education": ["Legacy MSc AI"],
            "roles": ["Legacy built NLP project"],
        }
    )

    text = build_cv_embedding_text(profile)

    assert "Legacy" not in text
    assert "Education:" not in text
    assert "Experience:" not in text


def test_vector_validation_dimension_nan_inf_and_normalization() -> None:
    assert validate_vector([1.0, 2.0], dimension=2) == [1.0, 2.0]
    with pytest.raises(ValueError, match="dimension"):
        validate_vector([1.0], dimension=2)
    with pytest.raises(ValueError, match="NaN"):
        validate_vector([math.nan, 1.0], dimension=2)
    with pytest.raises(ValueError, match="infinite"):
        validate_vector([math.inf, 1.0], dimension=2)
    normalized = prepare_vector([3.0, 4.0], dimension=2, normalize=True)
    assert normalized == pytest.approx([0.6, 0.8])
    assert vector_to_pg([0.6, 0.8]) == "[0.6,0.8]"


def test_embed_cv_profile_uses_fake_encoder_and_returns_384_normalized_values() -> None:
    vector = embed_cv_profile(_profile(), encoder=FakeEncoder(), settings=get_embedding_settings())

    assert len(vector) == 384
    assert vector[0] == pytest.approx(1.0)
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def _profile() -> CvProfile:
    document = ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text="SECRET RAW CV Python",
        page_count=None,
        pages=[DocumentPage(page=None, text="SECRET RAW CV Python", start_offset=0, end_offset=20)],
        file_hash="a" * 64,
        file_size=20,
    )
    versions = ExtractionVersions(
        parser_version=document.parser_version,
        matcher_version="matcher",
        ontology_version="ontology",
    )
    skill = CvSkill(
        canonical_skill="Python",
        ontology_skill_id=1,
        matched_alias="Python",
        extraction_method="deterministic",
        evidence=EvidenceSpan(
            evidence_text="Python",
            document_start=14,
            document_end=20,
        ),
        matcher_version="matcher",
        ontology_version="ontology",
        validation_status="deterministic",
        confidence=1.0,
    )
    return CvProfile(
        document=document,
        versions=versions,
        skills=[skill],
        candidate=CandidateAttributes(
            career_level="junior",
            preferred_domains=["Data Science"],
            programming_languages=["Python"],
            databases=["PostgreSQL"],
        ),
        education_detail=EducationDetail(
            education_level="master/mba",
            education_field="AI",
            institution="Example University",
            evidence=["MSc AI at Example University"],
        ),
        experience_entries=[
            ExperienceEntry(
                title="Research Assistant",
                employer="Example Lab",
                entry_type="job",
                evidence="Built an NLP project for internal tooling",
            ),
        ],
    )
