"""Tests for deterministic CV skill extraction."""

from __future__ import annotations

from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.profile import DocumentPage, ParsedDocument
from jobmarket.skills.matcher import MATCHER_VERSION


def test_cv_skill_extraction_english_aliases_and_offsets() -> None:
    document = _document("Senior Data Scientist with Python and AWS experience.")
    profile = extract_cv_profile(document)
    skills = {skill.canonical_skill: skill for skill in profile.skills}

    assert {"Data Science", "Python", "AWS"} <= set(skills)
    python = skills["Python"]
    assert python.matcher_version == MATCHER_VERSION
    assert document.text[python.evidence.document_start : python.evidence.document_end] == "Python"
    assert python.evidence.page is None


def test_cv_skill_extraction_french_aliases() -> None:
    document = _document("Ingénieur IA avec apprentissage automatique et sécurité informatique.")
    profile = extract_cv_profile(document)

    assert {"Artificial Intelligence", "Machine Learning", "Cybersecurity"} <= {
        skill.canonical_skill for skill in profile.skills
    }


def test_cv_skill_extraction_page_mapping() -> None:
    text = "Python AWS\n\nData Science"
    document = ParsedDocument(
        filename="cv.pdf",
        mime_type="application/pdf",
        text=text,
        page_count=2,
        pages=[
            DocumentPage(page=1, text="Python AWS", start_offset=0, end_offset=10),
            DocumentPage(page=2, text="Data Science", start_offset=12, end_offset=24),
        ],
        warnings=[],
        file_hash="a" * 64,
        file_size=10,
    )

    profile = extract_cv_profile(document)
    data_science = next(
        skill for skill in profile.skills if skill.canonical_skill == "Data Science"
    )

    assert data_science.evidence.page == 2
    assert data_science.evidence.page_start == 0
    assert data_science.evidence.page_end == 12
    assert document.pages[1].text[0:12] == data_science.evidence.evidence_text


def test_duplicate_canonical_skill_is_returned_once() -> None:
    profile = extract_cv_profile(_document("Python, python3, and PYTHON."))

    assert [skill.canonical_skill for skill in profile.skills].count("Python") == 1


def test_no_skill_cv_is_valid() -> None:
    profile = extract_cv_profile(_document("I enjoy stakeholder interviews and writing."))

    assert profile.skills == []


def test_cv_candidate_attributes_are_deterministic_and_non_invented() -> None:
    text = (
        "Student in Master Data Science. Stage de 6 months on Python, AWS, "
        "machine learning and PostgreSQL."
    )
    profile = extract_cv_profile(_document(text))

    assert profile.candidate.current_role == "student"
    assert profile.candidate.career_level == "student"
    assert profile.candidate.education_status == "Student"
    assert profile.candidate.experience.internship_count == 1
    assert profile.candidate.experience.total_months == 6
    assert profile.candidate.experience.total_years == 0.5
    assert "Data Science" in profile.candidate.preferred_domains
    assert "Machine Learning" in profile.candidate.preferred_domains
    assert "Python" in profile.candidate.programming_languages
    assert "AWS" in profile.candidate.cloud_platforms
    assert "PostgreSQL" in profile.candidate.databases


def test_candidate_experience_is_not_invented_without_explicit_duration() -> None:
    profile = extract_cv_profile(_document("Python AWS Data Science portfolio projects."))

    assert profile.candidate.experience.total_months is None
    assert profile.candidate.experience.total_years is None
    assert profile.candidate.career_level == "unknown"


def _document(text: str) -> ParsedDocument:
    return ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text=text,
        page_count=None,
        pages=[DocumentPage(page=None, text=text, start_offset=0, end_offset=len(text))],
        warnings=[],
        file_hash="a" * 64,
        file_size=len(text.encode()),
    )
