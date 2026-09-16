"""Tests for data-driven CV improvement. All LLM calls mocked — zero real network access."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from jobmarket.cv.improver import (
    CvImproverError,
    SkillGapEntry,
    generate_improved_cv,
)
from jobmarket.cv.profile import (
    CandidateAttributes,
    CvProfile,
    CvSkill,
    DocumentPage,
    EvidenceSpan,
    ExperienceEntry,
    ExperienceSummary,
    ExtractionQuality,
    ExtractionVersions,
    ParsedDocument,
    RoleInference,
)
from jobmarket.cv.workflow import RecommendationOutput

CV_TEXT = (
    "First Last\nData Scientist\n\n"
    "Experience\nData Scientist at Resume Worded, built ML pipelines with Python and SQL, "
    "improved model accuracy by 12%.\n\n"
    "Data Analyst at Polyhire, analyzed sales data using Python and Excel.\n"
)


def _document() -> ParsedDocument:
    return ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text=CV_TEXT,
        page_count=None,
        pages=[DocumentPage(page=None, text=CV_TEXT, start_offset=0, end_offset=len(CV_TEXT))],
        warnings=[],
        file_hash="a" * 64,
        file_size=len(CV_TEXT.encode()),
    )


def _skill(canonical: str, evidence_text: str) -> CvSkill:
    start = CV_TEXT.find(evidence_text)
    return CvSkill(
        canonical_skill=canonical,
        ontology_skill_id=1,
        matched_alias=evidence_text,
        extraction_method="deterministic",
        evidence=EvidenceSpan(
            evidence_text=evidence_text,
            document_start=start if start >= 0 else 0,
            document_end=(start + len(evidence_text)) if start >= 0 else len(evidence_text),
        ),
        matcher_version="v1",
        ontology_version="v1",
        validation_status="deterministic",
        confidence=1.0,
    )


def _experience(title: str, employer: str, evidence: str) -> ExperienceEntry:
    return ExperienceEntry(
        title=title,
        employer=employer,
        start_date=None,
        end_date=None,
        duration_months=None,
        entry_type="job",
        domain=None,
        confidence=0.9,
        evidence=evidence,
    )


def _profile(
    *,
    skills: list[CvSkill] | None = None,
    experiences: list[ExperienceEntry] | None = None,
    career_level: str = "mid",
    confidence_label: str = "high",
    abstention_reason: str | None = None,
) -> CvProfile:
    skills = skills if skills is not None else [
        _skill("Python", "Python"),
        _skill("SQL", "SQL"),
        _skill("Excel", "Excel"),
        _skill("Machine Learning", "ML"),
    ]
    experiences = (
        experiences
        if experiences is not None
        else [
            _experience(
                "Data Scientist",
                "Resume Worded",
                "Data Scientist at Resume Worded, built ML pipelines with Python and SQL, "
                "improved model accuracy by 12%.",
            ),
            _experience(
                "Data Analyst",
                "Polyhire",
                "Data Analyst at Polyhire, analyzed sales data using Python and Excel.",
            ),
        ]
    )
    return CvProfile(
        document=_document(),
        versions=ExtractionVersions(
            parser_version="v1", matcher_version="v1", ontology_version="v1"
        ),
        skills=skills,
        candidate=CandidateAttributes(
            career_level=career_level,
            experience=ExperienceSummary(
                professional_experience_count=len(experiences), total_years=3.0
            ),
        ),
        experience_entries=experiences,
        inferred_domains=[RoleInference(value="Data Science", confidence=0.6)],
        extraction_quality=ExtractionQuality(
            extraction_confidence_label=confidence_label,
            abstention_reason=abstention_reason,
        ),
    )


def _matched_job(job_id: int, title: str, matched_skills: list[str]) -> RecommendationOutput:
    return RecommendationOutput(
        job_id=job_id,
        title=title,
        company="Some Employer",
        final_score=80.0,
        skill_score=0.8,
        role_domain_score=0.8,
        career_level_compatibility=0.8,
        matched_skills=matched_skills,
        missing_important_skills=[],
        explanation="matched",
    )


def _mock_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


def _valid_payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "professional_summary": "Mid-level professional skilled in Python and SQL.",
        "experiences": [
            {"index": 0, "improved_description": "Built and shipped ML pipelines in Python/SQL."},
            {"index": 1, "improved_description": "Analyzed sales data using Python and Excel."},
        ],
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_generate_improved_cv_uses_llm_summary_and_rewords_experiences() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    profile = _profile()
    matched_jobs = [_matched_job(1, "ML Engineer", ["Python", "SQL"])]

    result = generate_improved_cv(profile, [], matched_jobs, client=client)

    assert result.improved_summary == "Mid-level professional skilled in Python and SQL."
    assert result.improved_experiences[0].improved_description == (
        "Built and shipped ML pipelines in Python/SQL."
    )
    assert result.dropped_ungrounded_mentions == []


def test_generate_improved_cv_drops_ungrounded_skill_in_summary() -> None:
    """A skill the candidate doesn't have must never reach the improved CV body."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            professional_summary="Expert in Python, SQL, and Kubernetes orchestration."
        )
    )
    profile = _profile()  # only Python + SQL grounded — no Kubernetes

    result = generate_improved_cv(profile, [], [], client=client)

    assert "Kubernetes" not in result.improved_summary
    assert result.improved_summary == (
        "Mid professional with grounded experience in Excel, Machine Learning, Python, SQL."
    )
    assert any("Kubernetes" in note for note in result.dropped_ungrounded_mentions)


def test_generate_improved_cv_drops_ungrounded_skill_in_experience_bullet() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "index": 0,
                    "improved_description": (
                        "Built ML pipelines in Python and deployed them with Docker and Kubernetes."
                    ),
                },
                {"index": 1, "improved_description": "Analyzed sales data using Python and Excel."},
            ]
        )
    )
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    # Docker/Kubernetes aren't grounded -> the whole bullet falls back to original evidence.
    assert result.improved_experiences[0].improved_description == (
        result.improved_experiences[0].original_evidence
    )
    assert any("experience[Resume Worded]" in note for note in result.dropped_ungrounded_mentions)


def test_own_job_title_domain_word_is_allowed_in_its_own_bullet() -> None:
    """Restating a job's own real title isn't fabrication, even if that word isn't
    in the formal skills list. Job 0's title is "Data Scientist", which the ontology
    resolves to canonical "Data Science" — mentioning that in job 0's own rewrite
    must be allowed, unlike an unrelated ungrounded word (see the test above)."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "index": 0,
                    "improved_description": (
                        "Applied data science techniques to build ML pipelines in Python/SQL."
                    ),
                },
                {"index": 1, "improved_description": "Analyzed sales data using Python and Excel."},
            ]
        )
    )
    profile = _profile()  # job 0 title == "Data Scientist" -> resolves to "Data Science"

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[0].improved_description == (
        "Applied data science techniques to build ML pipelines in Python/SQL."
    )
    assert result.dropped_ungrounded_mentions == []


def test_domain_word_from_a_different_jobs_title_is_still_rejected() -> None:
    """The same "Data Science" mention that's fine in job 0's own bullet (see test
    above) must NOT leak into job 1's bullet — job 1's title is "Data Analyst",
    which doesn't resolve to "Data Science", so it has no claim to that word."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "index": 0,
                    "improved_description": "Built and shipped ML pipelines in Python/SQL.",
                },
                {
                    "index": 1,
                    "improved_description": (
                        "Applied data science methods to analyze sales data using Python and Excel."
                    ),
                },
            ]
        )
    )
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[1].improved_description == (
        result.improved_experiences[1].original_evidence
    )
    assert any(
        "experience[Polyhire]" in note and "Data Science" in note
        for note in result.dropped_ungrounded_mentions
    )


def test_summary_may_reference_domain_word_from_any_real_job_title() -> None:
    """The summary isn't tied to one job, so its allowance is the union across all
    real job titles/evidence — "Data Science" is grounded via job 0's title even
    though the summary describes the whole profile."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            professional_summary=(
                "Mid-level professional with data science experience in Python and SQL."
            )
        )
    )
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_summary == (
        "Mid-level professional with data science experience in Python and SQL."
    )
    assert result.dropped_ungrounded_mentions == []


def test_generate_improved_cv_rejects_fabricated_number() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "index": 0,
                    "improved_description": (
                        "Built ML pipelines in Python/SQL, improving model accuracy by 45%."
                    ),
                },
                {"index": 1, "improved_description": "Analyzed sales data using Python and Excel."},
            ]
        )
    )
    profile = _profile()  # original says 12%, not 45%

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[0].improved_description == (
        result.improved_experiences[0].original_evidence
    )
    assert any("introduced a number" in note for note in result.dropped_ungrounded_mentions)


def test_generate_improved_cv_rejects_decimal_shifted_number() -> None:
    """A decimal-point shift (12% -> 1.2%) is a materially different number, not a
    reformatting — must be rejected just like a wholesale fabrication. Regression for
    a real bug: the original number check stripped "." during normalization, so "12"
    and "1.2" both collapsed to the same token and a decimal alteration slipped through
    ungrounded."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "index": 0,
                    "improved_description": (
                        "Built ML pipelines in Python/SQL, improving model accuracy by 1.2%."
                    ),
                },
                {"index": 1, "improved_description": "Analyzed sales data using Python and Excel."},
            ]
        )
    )
    profile = _profile()  # original says 12%, not 1.2%

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[0].improved_description == (
        result.improved_experiences[0].original_evidence
    )
    assert any("introduced a number" in note for note in result.dropped_ungrounded_mentions)


def test_generate_improved_cv_allows_comma_reformatted_number() -> None:
    """A thousands-separator reformatting (1234 -> 1,234) is the same real number,
    not a fabrication — must be allowed, unlike the decimal-shift case above."""
    client = MagicMock()
    profile = _profile(
        experiences=[
            _experience(
                "Data Engineer",
                "BigCo",
                "Data Engineer at BigCo, processed 1234 records daily using Python.",
            )
        ]
    )
    client.chat.completions.create.return_value = _mock_response(
        json.dumps(
            {
                "professional_summary": "Experienced professional.",
                "experiences": [
                    {
                        "index": 0,
                        "improved_description": (
                            "Processed 1,234 records daily as a Data Engineer using Python."
                        ),
                    }
                ],
            }
        )
    )

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[0].improved_description == (
        "Processed 1,234 records daily as a Data Engineer using Python."
    )
    assert result.dropped_ungrounded_mentions == []


def test_generate_improved_cv_rejects_cross_employer_contamination() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {"index": 0, "improved_description": "Built ML pipelines in Python and SQL."},
                {
                    "index": 1,
                    # Wrongly references job 0's employer inside job 1's bullet.
                    "improved_description": (
                        "Analyzed sales data using Python, similar to my work at Resume Worded."
                    ),
                },
            ]
        )
    )
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[1].improved_description == (
        result.improved_experiences[1].original_evidence
    )
    assert any("different employer" in note for note in result.dropped_ungrounded_mentions)


def test_generate_improved_cv_preserves_facts_regardless_of_llm_order() -> None:
    """Company/title/dates are never sourced from the LLM — index-based reattachment
    means even an out-of-order or partial LLM response can't corrupt them."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(
            experiences=[
                {
                    "index": 1,
                    "improved_description": "Analyzed sales data using Python and Excel.",
                },
                {
                    "index": 0,
                    "improved_description": "Built and shipped ML pipelines in Python/SQL.",
                },
            ]
        )
    )
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[0].title == "Data Scientist"
    assert result.improved_experiences[0].employer == "Resume Worded"
    assert result.improved_experiences[1].title == "Data Analyst"
    assert result.improved_experiences[1].employer == "Polyhire"
    # Both still got their (correctly index-matched) reword despite arriving out of order.
    assert result.improved_experiences[0].improved_description == (
        "Built and shipped ML pipelines in Python/SQL."
    )
    assert result.improved_experiences[1].improved_description == (
        "Analyzed sales data using Python and Excel."
    )


def test_generate_improved_cv_missing_index_falls_back_to_original() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _valid_payload(experiences=[{"index": 0, "improved_description": "Reworded job 0."}])
    )
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences[1].improved_description == (
        result.improved_experiences[1].original_evidence
    )


def test_skills_section_is_grounded_profile_skills_only() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.skills_section == ["Excel", "Machine Learning", "Python", "SQL"]


def test_recommended_skills_pull_from_skill_gap_with_correct_demand() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    profile = _profile()
    skill_gap = [
        SkillGapEntry(
            canonical_skill="Docker",
            category="tool",
            missing_in_top_matches=24,
            overall_job_demand=4800,
        ),
        SkillGapEntry(
            canonical_skill="Kubernetes",
            category="tool",
            missing_in_top_matches=10,
            overall_job_demand=2000,
        ),
    ]
    matched_jobs = [_matched_job(i, f"Job {i}", ["Python"]) for i in range(50)]

    result = generate_improved_cv(profile, skill_gap, matched_jobs, client=client)

    by_skill = {item.canonical_skill: item for item in result.recommended_skills_to_develop}
    assert by_skill["Docker"].demand_pct_of_matches == 48.0
    assert by_skill["Docker"].overall_job_demand == 4800
    assert by_skill["Kubernetes"].demand_pct_of_matches == 20.0
    # Recommended skills are never folded into the grounded skills section.
    assert "Docker" not in result.skills_section
    assert "Kubernetes" not in result.skills_section


def test_degraded_extraction_surfaces_warning_but_still_generates() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    profile = _profile(confidence_label="low", abstention_reason="Only 1 grounded skill found.")

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.source_degraded is True
    assert result.source_degraded_reason == "Only 1 grounded skill found."
    # Degraded doesn't mean empty/blocked — the CV still gets improved, just flagged.
    assert result.improved_summary != ""


def test_degraded_extraction_default_reason_when_none_given() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    profile = _profile(confidence_label="low", abstention_reason=None)

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.source_degraded is True
    assert result.source_degraded_reason is not None
    assert "low" in result.source_degraded_reason.lower()


def test_high_confidence_extraction_is_not_flagged_degraded() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_valid_payload())
    profile = _profile(confidence_label="high")

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.source_degraded is False
    assert result.source_degraded_reason is None


def test_call_llm_cv_improvement_retries_once_on_malformed_json() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _mock_response("not json"),
        _mock_response(_valid_payload()),
    ]
    profile = _profile()

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_summary != ""
    assert client.chat.completions.create.call_count == 2


def test_generate_improved_cv_raises_when_llm_fails_outright() -> None:
    """No deterministic fallback exists for improvement — a total LLM failure must be
    surfaced honestly, never silently replaced with a fabricated 'improved' CV."""
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("network down")
    profile = _profile()

    try:
        generate_improved_cv(profile, [], [], client=client)
        raise AssertionError("expected CvImproverError")
    except CvImproverError as exc:
        assert "network down" in str(exc)


def test_generate_improved_cv_with_no_experience_entries() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        json.dumps({"professional_summary": "Skilled in Python and SQL.", "experiences": []})
    )
    profile = _profile(experiences=[])

    result = generate_improved_cv(profile, [], [], client=client)

    assert result.improved_experiences == []
    assert result.improved_summary == "Skilled in Python and SQL."
