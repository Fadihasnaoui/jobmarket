"""Tests for the CV Quality Report. All LLM calls mocked — zero real network access."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from jobmarket.cv.improver import SkillGapEntry
from jobmarket.cv.profile import (
    CandidateAttributes,
    CvProfile,
    CvSection,
    CvSkill,
    DocumentPage,
    EvidenceSpan,
    ExperienceEntry,
    ExtractionQuality,
    ExtractionVersions,
    ParsedDocument,
)
from jobmarket.cv.quality_report import (
    ContactSignals,
    detect_contact_signals,
    enrich_experience_impact_evidence,
    generate_cv_quality_report,
)
from jobmarket.cv.workflow import RecommendationOutput

CV_TEXT = "First Last\nData Scientist\n\nExperience\nBuilt pipelines. Improved accuracy by 12%.\n"


def _document() -> ParsedDocument:
    return ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text=CV_TEXT,
        pages=[DocumentPage(page=None, text=CV_TEXT, start_offset=0, end_offset=len(CV_TEXT))],
        warnings=[],
        file_hash="a" * 64,
        file_size=len(CV_TEXT.encode()),
    )


def _skill(canonical: str) -> CvSkill:
    return CvSkill(
        canonical_skill=canonical,
        ontology_skill_id=1,
        matched_alias=canonical,
        extraction_method="deterministic",
        evidence=EvidenceSpan(
            evidence_text=canonical, document_start=0, document_end=len(canonical)
        ),
        matcher_version="v1",
        ontology_version="v1",
        validation_status="deterministic",
        confidence=1.0,
    )


def _experience(
    title: str, employer: str, evidence: str, *, impact_evidence: str | None = None
) -> ExperienceEntry:
    return ExperienceEntry(
        title=title,
        employer=employer,
        entry_type="job",
        confidence=0.9,
        evidence=evidence,
        impact_evidence=impact_evidence,
    )


def _profile(
    *,
    skills: list[CvSkill] | None = None,
    experiences: list[ExperienceEntry] | None = None,
    sections: list[CvSection] | None = None,
    confidence_label: str = "high",
    abstention_reason: str | None = None,
) -> CvProfile:
    skills = skills if skills is not None else [_skill("Python"), _skill("SQL")]
    experiences = (
        experiences
        if experiences is not None
        else [
            _experience(
                "Data Scientist",
                "Resume Worded",
                "Built Python/SQL pipelines; improved accuracy by 12%.",
            ),
        ]
    )
    return CvProfile(
        document=_document(),
        versions=ExtractionVersions(
            parser_version="v1", matcher_version="v1", ontology_version="v1"
        ),
        skills=skills,
        candidate=CandidateAttributes(career_level="mid"),
        experience_entries=experiences,
        sections=sections or [],
        extraction_quality=ExtractionQuality(
            extraction_confidence_label=confidence_label, abstention_reason=abstention_reason
        ),
    )


def _matched_job(job_id: int) -> RecommendationOutput:
    return RecommendationOutput(
        job_id=job_id,
        title=f"Job {job_id}",
        company="Some Employer",
        final_score=80.0,
        skill_score=0.8,
        role_domain_score=0.8,
        career_level_compatibility=0.8,
        matched_skills=["Python"],
        missing_important_skills=[],
        explanation="matched",
    )


def _mock_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=None)


def _spelling_payload(issues: list[dict[str, object]]) -> str:
    return json.dumps({"issues": issues})


def test_spelling_check_accepts_a_real_correction() -> None:
    client = MagicMock()
    profile = _profile(
        experiences=[
            _experience(
                "Data Scientist",
                "Resume Worded",
                "Buillt pipelines in Python and SQL, improved model accuracy by 12%.",
            )
        ]
    )
    fixed_text = "Built pipelines in Python and SQL, improved model accuracy by 12%."
    client.chat.completions.create.return_value = _mock_response(
        _spelling_payload([{"index": 0, "corrected": fixed_text}])
    )

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert len(report.spelling_issues) == 1
    assert report.spelling_issues[0].corrected == fixed_text
    check = next(c for c in report.checks if c.id == "spelling_grammar")
    assert check.status in ("warning", "fail")  # 1 issue found



def test_spelling_check_ignores_a_whitespace_only_change() -> None:
    client = MagicMock()
    profile = _profile(
        experiences=[
            _experience("Developer", "Acme", "Built  Python services and improved uptime by 12%.")
        ]
    )
    client.chat.completions.create.return_value = _mock_response(
        _spelling_payload(
            [{"index": 0, "corrected": "Built Python services and improved uptime by 12%."}]
        )
    )

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert report.spelling_issues == []
    check = next(item for item in report.checks if item.id == "spelling_grammar")
    assert check.status == "pass"


def test_spelling_check_rejects_fix_that_changes_a_number() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _spelling_payload(
            [
                {
                    "index": 0,
                    "corrected": "Built Python/SQL pipelines; improved accuracy by 1.2%.",
                }
            ]
        )
    )
    profile = _profile()  # original says 12%, not 1.2%

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert report.spelling_issues == []
    assert any("changed a number" in note for note in report.dropped_spelling_suggestions)


def test_spelling_check_rejects_fix_that_changes_a_skill() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _spelling_payload(
            [
                {
                    "index": 0,
                    "corrected": "Built Python/SQL/Kubernetes pipelines; improved accuracy by 12%.",
                }
            ]
        )
    )
    profile = _profile()

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert report.spelling_issues == []
    assert any("skill mention" in note for note in report.dropped_spelling_suggestions)


def test_spelling_check_rejects_disguised_rewrite() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(
        _spelling_payload(
            [
                {
                    "index": 0,
                    "corrected": (
                        "Spearheaded the design, development, and deployment of robust data "
                        "pipelines leveraging Python and SQL, resulting in a substantial 12% "
                        "improvement in overall model accuracy across the board."
                    ),
                }
            ]
        )
    )
    profile = _profile()

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert report.spelling_issues == []
    assert any("rewrite" in note for note in report.dropped_spelling_suggestions)


def test_spelling_check_not_checked_on_llm_failure_does_not_tank_score() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = ConnectionError("network down")
    profile = _profile()

    report = generate_cv_quality_report(profile, [], [], client=client)

    spelling = next(c for c in report.checks if c.id == "spelling_grammar")
    assert spelling.status == "not_checked"
    # Score is rescaled over the checks that DID run, not zeroed out for this one.
    assert report.overall_score > 0


def test_impact_quantification_flags_unquantified_bullet() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(
        experiences=[
            _experience("Sales Rep", "Acme", "Increased sales and grew the customer base."),
        ]
    )

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert len(report.unquantified_bullets) == 1
    check = next(c for c in report.checks if c.id == "impact_quantification")
    assert check.status == "fail"


def test_impact_quantification_passes_with_quantified_bullet() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()  # default bullet has "12%"

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert report.unquantified_bullets == []
    check = next(c for c in report.checks if c.id == "impact_quantification")
    assert check.status == "pass"



def test_impact_quantification_uses_grounded_per_role_impact_evidence() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(
        experiences=[
            _experience(
                "Tech Mentor",
                "Code For Good",
                "Tech Mentor Code For Good",
                impact_evidence=(
                    "Volunteered as a tech mentor, guiding young enthusiasts in coding projects."
                ),
            ),
            _experience(
                "Coding Bootcamp Participant",
                "TechLabs Academy",
                "Coding Bootcamp Participant TechLabs Academy",
                impact_evidence=(
                    "Built an application, resulting in a 50% improvement in learning efficiency."
                ),
            ),
            _experience(
                "Student Developer Volunteer",
                "Girl Develop It",
                "Student Developer Volunteer Girl Develop It",
                impact_evidence="Reduced bugs and errors by 40% while supporting students.",
            ),
        ]
    )

    report = generate_cv_quality_report(profile, [], [], client=client)

    assert [item.source_label for item in report.unquantified_bullets] == [
        "Tech Mentor @ Code For Good"
    ]
    check = next(item for item in report.checks if item.id == "impact_quantification")
    assert check.message == "2 of 3 experience entries include a measurable result."



def test_upload_time_impact_enrichment_recovers_metrics_without_llm_quote() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(
        experiences=[
            _experience("Tech Mentor", "Code For Good", "Tech Mentor Code For Good"),
            _experience(
                "Coding Bootcamp Participant",
                "TechLabs Academy",
                "Coding Bootcamp Participant TechLabs Academy",
            ),
            _experience(
                "Student Developer Volunteer",
                "Girl Develop It",
                "Student Developer Volunteer Girl Develop It",
            ),
        ]
    )
    raw_cv_text = """Tech Mentor
Code For Good
Volunteered as a tech mentor for coding workshops.

Coding Bootcamp Participant
TechLabs Academy
Built a web application, resulting in a 50% improvement in learning efficiency.

Student Developer Volunteer
Girl Develop It
Reduced bugs and errors by 40% while supporting students.
"""

    enriched = enrich_experience_impact_evidence(profile, raw_cv_text)
    report = generate_cv_quality_report(enriched, [], [], client=client)

    assert "50%" in (enriched.experience_entries[1].impact_evidence or "")
    assert "40%" in (enriched.experience_entries[2].impact_evidence or "")
    assert [item.source_label for item in report.unquantified_bullets] == [
        "Tech Mentor @ Code For Good"
    ]
    check = next(item for item in report.checks if item.id == "impact_quantification")
    assert check.message == "2 of 3 experience entries include a measurable result."


def test_missing_skills_section_flagged() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(skills=[])

    report = generate_cv_quality_report(profile, [], [], client=client)

    check = next(c for c in report.checks if c.id == "missing_sections")
    assert "missing" in check.message.lower()
    assert check.status in ("warning", "fail")


def test_summary_presence_unknown_when_no_sections_data() -> None:
    """LLM extraction path never populates `sections` — must report "not verifiable",
    never silently claim it's missing."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(sections=[])

    report = generate_cv_quality_report(profile, [], [], client=client)

    check = next(c for c in report.checks if c.id == "missing_sections")
    assert "could not be verified" in check.message.lower()


def test_summary_detected_from_sections() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(
        sections=[CvSection(label="Summary", normalized_label="summary", start=0, end=10)]
    )

    report = generate_cv_quality_report(profile, [], [], client=client)

    check = next(c for c in report.checks if c.id == "missing_sections")
    assert "a professional summary section was detected" in check.message.lower()


def test_contact_check_name_plus_form_contact_passes() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()

    report = generate_cv_quality_report(
        profile,
        [],
        [],
        candidate_name="Jane Doe",
        contact_info="jane@example.com | +1 555 0100 | linkedin.com/in/jane",
        client=client,
    )

    check = next(c for c in report.checks if c.id == "contact_info")
    assert check.status == "pass"
    assert check.score == check.max_score


def test_contact_check_reads_real_cv_text_not_just_the_form() -> None:
    """The bug this check exists to fix: a CV with real email/phone/LinkedIn must not
    be flagged as missing contact info just because the optional form was left blank."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()
    signals = ContactSignals(has_email=True, has_phone=True, has_linkedin_or_github=True)

    report = generate_cv_quality_report(
        profile, [], [], contact_signals=signals, client=client
    )

    check = next(c for c in report.checks if c.id == "contact_info")
    assert check.status == "warning"  # all 3 channels found, but no name was provided
    assert "found in your cv" in check.message.lower()
    assert "email" not in (check.suggestion or "").lower()


def test_contact_check_flags_only_the_channels_genuinely_absent() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()
    signals = ContactSignals(has_email=False, has_phone=True, has_linkedin_or_github=True)

    report = generate_cv_quality_report(
        profile, [], [], candidate_name="Jane Doe", contact_signals=signals, client=client
    )

    check = next(c for c in report.checks if c.id == "contact_info")
    assert check.status == "warning"
    assert "email" in check.message.lower()
    assert "phone" not in check.message.lower()
    assert "linkedin" not in check.message.lower()


def test_contact_check_neither_provided_fails_with_suggestion() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()

    report = generate_cv_quality_report(profile, [], [], client=client)

    check = next(c for c in report.checks if c.id == "contact_info")
    assert check.status == "fail"
    assert check.score == 0.0
    assert check.suggestion is not None
    assert "never stored" in check.suggestion.lower()


def test_detect_contact_signals_finds_real_patterns() -> None:
    text = "Contact: jane.doe@example.com, +1 (555) 010-0199, linkedin.com/in/janedoe"
    signals = detect_contact_signals(text)
    assert signals.has_email is True
    assert signals.has_phone is True
    assert signals.has_linkedin_or_github is True



def test_contact_check_recognises_a_name_in_the_cv_header() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    signals = detect_contact_signals("LUNA THOMAS\n+1-614-555-1234 | Email | LinkedIn")

    assert signals.has_name is True
    report = generate_cv_quality_report(_profile(), [], [], contact_signals=signals, client=client)

    check = next(item for item in report.checks if item.id == "contact_info")
    assert check.status == "warning"
    assert "email" in check.message.lower()
    assert "phone" not in check.message.lower()
    assert "linkedin" not in check.message.lower()
    assert "no name was provided" not in check.message.lower()


def test_detect_contact_signals_ignores_year_ranges_and_bare_text() -> None:
    text = "Experience 2020-2023. Education 2018-2020. No links or numbers here."
    signals = detect_contact_signals(text)
    assert signals.has_email is False
    assert signals.has_phone is False
    assert signals.has_linkedin_or_github is False


def test_detect_contact_signals_accepts_bare_linkedin_mention() -> None:
    """A hyperlinked 'LinkedIn' anchor loses its href during plain-text PDF
    extraction, leaving only the bare word — this must still count as present."""
    signals = detect_contact_signals("References available. LinkedIn | GitHub")
    assert signals.has_linkedin_or_github is True


def test_ats_check_flags_low_confidence() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(confidence_label="low", abstention_reason="Only 1 grounded skill found.")

    report = generate_cv_quality_report(profile, [], [], client=client)

    check = next(c for c in report.checks if c.id == "ats_readiness")
    assert "high-confidence extraction" in check.message
    assert report.source_degraded is True
    assert report.source_degraded_reason == "Only 1 grounded skill found."


def test_market_skill_gap_check_reflects_gap_count() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()
    skill_gap = [
        SkillGapEntry(
            canonical_skill="Docker",
            category="tool",
            missing_in_top_matches=5,
            overall_job_demand=1000,
        ),
        SkillGapEntry(
            canonical_skill="Kubernetes",
            category="tool",
            missing_in_top_matches=3,
            overall_job_demand=500,
        ),
    ]
    matched_jobs = [_matched_job(i) for i in range(10)]

    report = generate_cv_quality_report(profile, skill_gap, matched_jobs, client=client)

    assert len(report.recommended_skills_to_develop) == 2
    by_skill = {item.canonical_skill: item for item in report.recommended_skills_to_develop}
    assert by_skill["Docker"].demand_pct_of_matches == 50.0
    check = next(c for c in report.checks if c.id == "market_skill_gap")
    assert "Docker" in check.message


def test_overall_score_is_within_bounds() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile()

    report = generate_cv_quality_report(
        profile, [], [], candidate_name="Jane Doe", contact_info="jane@example.com", client=client
    )

    assert 0 <= report.overall_score <= 100


def test_no_experience_entries_scores_zero_on_impact() -> None:
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(_spelling_payload([]))
    profile = _profile(experiences=[])

    report = generate_cv_quality_report(profile, [], [], client=client)

    check = next(c for c in report.checks if c.id == "impact_quantification")
    assert check.status == "fail"
    assert check.score == 0.0
