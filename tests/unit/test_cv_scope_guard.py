"""Tests for the out-of-scope (non-tech profile) guard."""

from __future__ import annotations

from jobmarket.cv.profile import (
    CandidateAttributes,
    CvProfile,
    CvSkill,
    DocumentPage,
    EvidenceSpan,
    ExtractionVersions,
    ParsedDocument,
    RoleInference,
)
from jobmarket.cv.scope_guard import OUT_OF_SCOPE_MESSAGE, out_of_scope_reason


def _document() -> ParsedDocument:
    return ParsedDocument(
        filename="cv.txt",
        mime_type="text/plain",
        text="x",
        page_count=None,
        pages=[DocumentPage(page=None, text="x", start_offset=0, end_offset=1)],
        warnings=[],
        file_hash="a" * 64,
        file_size=1,
    )


def _profile(
    *,
    skills: list[CvSkill] | None = None,
    domains: list[str] | None = None,
    roles: list[str] | None = None,
) -> CvProfile:
    document = _document()
    candidate = CandidateAttributes(preferred_domains=domains or [])
    return CvProfile(
        document=document,
        versions=ExtractionVersions(
            parser_version=document.parser_version, matcher_version="v", ontology_version="v"
        ),
        skills=skills or [],
        candidate=candidate,
        inferred_roles=[
            RoleInference(value=r, confidence=0.6, evidence=["x"]) for r in roles or []
        ],
    )


def _skill(canonical: str) -> CvSkill:
    return CvSkill(
        canonical_skill=canonical,
        ontology_skill_id=1,
        matched_alias=canonical,
        extraction_method="deterministic",
        evidence=EvidenceSpan(evidence_text="x", document_start=0, document_end=1),
        matcher_version="v",
        ontology_version="v",
        validation_status="deterministic",
        confidence=1.0,
    )


def test_fires_for_zero_tech_skills_and_business_domain() -> None:
    profile = _profile(domains=["Sales", "Customer Service"], roles=["Sales"])

    assert out_of_scope_reason(profile) == OUT_OF_SCOPE_MESSAGE


def test_fires_for_zero_tech_skills_and_non_tech_role_only() -> None:
    profile = _profile(domains=[], roles=["Management"])

    assert out_of_scope_reason(profile) == OUT_OF_SCOPE_MESSAGE


def test_does_not_fire_when_any_real_tech_skill_present() -> None:
    profile = _profile(skills=[_skill("Python")], domains=["Sales"], roles=["Sales"])

    assert out_of_scope_reason(profile) is None


def test_soft_skill_category_alone_does_not_count_as_tech_evidence() -> None:
    # "Client Relationship Management" is ontology category=soft_skill: common to
    # sales/support roles too, must not by itself prove a technical profile.
    profile = _profile(
        skills=[_skill("Client Relationship Management")],
        domains=["Sales", "Customer Service"],
        roles=["Sales"],
    )

    assert out_of_scope_reason(profile) == OUT_OF_SCOPE_MESSAGE


def test_does_not_fire_when_technical_domain_present() -> None:
    profile = _profile(domains=["AI", "Sales"], roles=["Sales"])

    assert out_of_scope_reason(profile) is None


def test_does_not_fire_on_domain_role_silence_alone() -> None:
    # No domain/role signal at all is the low-confidence warning's job, not this guard.
    profile = _profile(domains=[], roles=[])

    assert out_of_scope_reason(profile) is None


def test_thin_tech_profile_with_two_skills_is_never_out_of_scope() -> None:
    profile = _profile(skills=[_skill("Python"), _skill("SQL")], domains=[], roles=[])

    assert out_of_scope_reason(profile) is None
