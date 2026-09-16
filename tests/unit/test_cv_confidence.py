"""Tests for cautious-scoring dampening applied to low-confidence CV profiles."""

from __future__ import annotations

from jobmarket.cv.confidence import LOW_CONFIDENCE_CAUTION_PENALTY, apply_confidence_caution
from jobmarket.cv.profile import (
    CvProfile,
    DocumentPage,
    ExtractionQuality,
    ExtractionVersions,
    JobRecommendation,
    ParsedDocument,
    ScoreComponents,
)


def _profile(label: str, score: float = 0.3) -> CvProfile:
    document = ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text="x",
        page_count=None,
        pages=[DocumentPage(page=None, text="x", start_offset=0, end_offset=1)],
        warnings=[],
        file_hash="a" * 64,
        file_size=1,
    )
    return CvProfile(
        document=document,
        versions=ExtractionVersions(
            parser_version=document.parser_version,
            matcher_version="v",
            ontology_version="v",
        ),
        skills=[],
        extraction_quality=ExtractionQuality(
            extraction_confidence_score=score,
            extraction_confidence_label=label,  # type: ignore[arg-type]
        ),
    )


def _recommendation(job_id: int, score: float) -> JobRecommendation:
    return JobRecommendation(
        job_id=job_id,
        title="Job",
        company="Co",
        location=None,
        final_score=score,
        coverage_score=1.0,
        cv_overlap_score=1.0,
        cv_skill_count=1,
        job_skill_count=1,
        matched_skill_count=1,
        matched_skills=["Python"],
        missing_skills=[],
        enrichment_run_id=1,
        scoring_version="v",
        explanation="x",
        score_components=ScoreComponents(
            skill_score=1.0,
            experience_score=1.0,
            career_level_score=1.0,
            job_type_score=1.0,
            domain_score=1.0,
            location_score=1.0,
            weighted_score_before_penalty=1.0,
            penalty_total=0.0,
        ),
    )


def test_high_confidence_recommendations_are_untouched() -> None:
    recs = [_recommendation(1, 80.0)]

    result = apply_confidence_caution(recs, _profile("high"))

    assert result == recs


def test_low_confidence_dampens_score_and_tags_penalty() -> None:
    recs = [_recommendation(1, 80.0)]

    result = apply_confidence_caution(recs, _profile("low"))

    assert result[0].final_score == 65.0
    assert result[0].penalties[-1].code == "low_extraction_confidence"
    assert result[0].penalties[-1].amount == LOW_CONFIDENCE_CAUTION_PENALTY


def test_low_confidence_never_produces_negative_score() -> None:
    recs = [_recommendation(1, 5.0)]

    result = apply_confidence_caution(recs, _profile("low"))

    assert result[0].final_score == 0.0


def test_low_confidence_reranks_after_dampening() -> None:
    recs = [_recommendation(1, 50.0), _recommendation(2, 49.0)]

    result = apply_confidence_caution(recs, _profile("low"))

    assert [item.job_id for item in result] == [1, 2]
    assert [item.final_score for item in result] == [35.0, 34.0]


def test_empty_recommendations_returns_empty() -> None:
    assert apply_confidence_caution([], _profile("low")) == []
