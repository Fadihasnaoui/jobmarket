"""Deterministic evaluation fixtures for CV recommendation quality."""

from __future__ import annotations

import pytest

from jobmarket.cv.attributes import extract_candidate_attributes, extract_job_attributes
from jobmarket.cv.matching import (
    JOB_MATCHING_FORMULA,
    SCORE_WEIGHTS,
    SKILL_SCORE_WEIGHTS,
    _location_score,
    skill_profile_confidence,
    skill_specificity,
)
from jobmarket.cv.profile import CandidateAttributes, JobAttributes
from jobmarket.skills.ontology import load_ontology


def test_candidate_attribute_evaluation_fixtures() -> None:
    ontology = load_ontology()
    cases = [
        (
            "student",
            "Student Master Data Science, stage 6 months. Python Machine Learning.",
            {"Python", "Machine Learning", "Data Science"},
            "student",
            0.5,
        ),
        (
            "junior_data_scientist",
            "Junior Data Scientist with 2 years experience in Python and SQL.",
            {"Python", "SQL", "Data Science"},
            "junior",
            2.0,
        ),
        (
            "mid_backend",
            "Backend Developer with 3 years experience using Python and PostgreSQL.",
            {"Python", "PostgreSQL", "Backend Development"},
            "mid",
            3.0,
        ),
        (
            "senior_ai_engineer",
            "Senior AI Engineer, professional experience 6 years, AWS and ML.",
            {"AWS", "Machine Learning", "Artificial Intelligence"},
            "senior",
            6.0,
        ),
    ]

    results = []
    for name, text, skills, expected_level, expected_years in cases:
        attributes = extract_candidate_attributes(text, skills, ontology)
        results.append((name, attributes.career_level == expected_level))
        assert attributes.career_level == expected_level
        assert attributes.experience.total_years == pytest.approx(expected_years)

    assert sum(correct for _, correct in results) == len(cases)


def test_job_attribute_evaluation_fixtures() -> None:
    cases = [
        (
            "Senior Data Scientist",
            "Requires minimum 5 years experience. Hybrid CDI data science role.",
            "CDI",
            False,
            {"Data Science", "Python"},
            "senior",
            5.0,
            {"cdi", "hybrid", "on_site"},
        ),
        (
            "Stage Machine Learning",
            "Stage 6 months remote for ML prototypes.",
            "Stage",
            True,
            {"Machine Learning", "Python"},
            "internship",
            None,
            {"stage", "remote"},
        ),
        (
            "Freelance Data Engineer",
            "Mission freelance 2-4 years, cloud data pipelines.",
            "Freelance",
            None,
            {"Data Engineering", "AWS"},
            "unknown",
            2.0,
            {"freelance"},
        ),
        (
            "Lead Backend Developer",
            "Tech Lead, at least 7 years, on-site backend platform.",
            "CDI",
            False,
            {"Backend Development", "Python"},
            "lead",
            7.0,
            {"cdi", "on_site"},
        ),
    ]

    for title, description, contract, remote, skills, level, min_years, job_types in cases:
        attributes = extract_job_attributes(
            title=title,
            description=description,
            contract_type=contract,
            is_remote=remote,
            country="FR",
            city="Paris",
            skill_canonicals=skills,
        )
        assert attributes.career_level == level
        assert attributes.experience_min_years == min_years
        assert set(attributes.job_types) == job_types


def test_job_type_provenance_and_apprentissage_automatique_correction() -> None:
    title_attrs = extract_job_attributes(
        title="Alternance Data Scientist",
        description="Python machine learning role.",
        contract_type=None,
        is_remote=None,
        country="fr",
        city="Paris",
        skill_canonicals={"Data Science", "Machine Learning"},
    )
    assert "apprenticeship" in title_attrs.job_types
    assert title_attrs.job_type_provenance[0].source == "title"

    description_attrs = extract_job_attributes(
        title="Data Scientist",
        description="Contrat d'alternance pour projets ML.",
        contract_type=None,
        is_remote=None,
        country="fr",
        city="Paris",
        skill_canonicals={"Data Science", "Machine Learning"},
    )
    assert "apprenticeship" in description_attrs.job_types
    assert any(item.source == "description" for item in description_attrs.job_type_provenance)

    false_positive_attrs = extract_job_attributes(
        title="Ing?nieur IA - Data Scientist H/F",
        description="Concevoir des mod?les d'apprentissage automatique.",
        contract_type=None,
        is_remote=None,
        country="fr",
        city="Nantes",
        skill_canonicals={"Artificial Intelligence", "Data Science", "Machine Learning"},
    )
    assert false_positive_attrs.career_level == "unknown"
    assert "apprenticeship" not in false_positive_attrs.job_types
    assert false_positive_attrs.job_type_provenance == []


def test_skill_specificity_and_richness_rules() -> None:
    assert skill_specificity("Artificial Intelligence") == "generic"
    assert skill_specificity("Data Science") == "generic"
    assert skill_specificity("Retrieval-Augmented Generation") == "specialized"
    assert skill_specificity("PostgreSQL") == "specialized"
    assert skill_specificity("Python") == "standard"
    assert skill_profile_confidence(1) == pytest.approx(0.60)
    assert skill_profile_confidence(2) == pytest.approx(0.75)
    assert skill_profile_confidence(4) == pytest.approx(0.90)
    assert skill_profile_confidence(6) == pytest.approx(1.00)


def test_location_scoring_states() -> None:
    base_job = JobAttributes(country="fr", city="Paris")
    exact = CandidateAttributes(preferred_country="fr", preferred_city="Paris")
    remote = CandidateAttributes(remote_preference=True)
    unknown = CandidateAttributes()
    mismatch = CandidateAttributes(preferred_country="us", preferred_city="New York")

    assert _location_score(_profile_with_candidate(exact), base_job) == (1.0, "exact_match")
    assert _location_score(_profile_with_candidate(unknown), base_job) == (
        0.5,
        "unknown_candidate_preference",
    )
    assert _location_score(_profile_with_candidate(mismatch), base_job) == (0.25, "mismatch")
    assert _location_score(
        _profile_with_candidate(remote),
        JobAttributes(remote_mode="remote"),
    ) == (0.9, "remote_compatible")


def test_scoring_formula_weights_are_centralized_and_normalized() -> None:
    assert SCORE_WEIGHTS == {
        "skills": 0.50,
        "experience": 0.15,
        "career_level": 0.15,
        "domain": 0.10,
        "job_type": 0.05,
        "location": 0.05,
    }
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    assert sum(SKILL_SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    assert "penalty_total" in JOB_MATCHING_FORMULA
    assert "0.05*job_type_score" in JOB_MATCHING_FORMULA


def _profile_with_candidate(candidate: CandidateAttributes):
    from jobmarket.cv.profile import CvProfile, ExtractionVersions, ParsedDocument

    document = ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text="",
        page_count=None,
        pages=[],
        warnings=[],
        file_hash="a" * 64,
        file_size=0,
    )
    return CvProfile(
        document=document,
        versions=ExtractionVersions(
            parser_version=document.parser_version,
            matcher_version="matcher",
            ontology_version="ontology",
        ),
        skills=[],
        candidate=candidate,
    )
