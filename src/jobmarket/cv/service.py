"""High-level in-memory CV matching workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobmarket.cv.attributes import extract_candidate_attributes
from jobmarket.cv.confidence import apply_confidence_caution
from jobmarket.cv.documents import parse_document
from jobmarket.cv.extraction import load_skill_ids
from jobmarket.cv.llm_extraction import extract_cv_profile_with_fallback
from jobmarket.cv.llm_guard import validate_llm_skill_candidates
from jobmarket.cv.matching import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RECOMMENDATION_RUN_ID,
    RecommendationMode,
    recommend_jobs_for_cv,
    validation_run_warning,
)
from jobmarket.cv.profile import (
    CandidateAttributes,
    CvMatchResult,
    ExperienceSummary,
    ExtractionVersions,
)
from jobmarket.cv.scope_guard import out_of_scope_reason
from jobmarket.db.session import async_session_factory
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.skills.ontology import load_ontology


async def match_cv_document(
    path: str | Path,
    *,
    top: int = 20,
    run_id: int = DEFAULT_RECOMMENDATION_RUN_ID,
    guarded_candidates: list[dict[str, Any]] | str | None = None,
    mode: RecommendationMode = "lexical",
    alpha: float = DEFAULT_HYBRID_ALPHA,
    encoder: EmbeddingEncoder | None = None,
) -> CvMatchResult:
    """Parse a CV, extract grounded skills, and rank jobs without persistence."""
    document = parse_document(path)
    ontology = load_ontology()
    async with async_session_factory() as session:
        skill_ids = await load_skill_ids(session, ontology)
        extraction_result = extract_cv_profile_with_fallback(
            document, ontology=ontology, skill_ids=skill_ids
        )
        profile = extraction_result.profile
        llm_skill_candidates = extraction_result.skill_candidates
        fallback_reason = extraction_result.fallback_reason
        candidates = guarded_candidates if guarded_candidates is not None else llm_skill_candidates
        guard_report = validate_llm_skill_candidates(
            document,
            candidates,
            deterministic_skills=profile.skills,
            ontology=ontology,
            skill_ids=skill_ids,
        )
        final_candidate = _merge_candidate_attributes(
            profile.candidate,
            extract_candidate_attributes(
                document.text,
                {skill.canonical_skill for skill in guard_report.final_skills},
                ontology,
            ),
        )
        final_profile = profile.model_copy(
            update={
                "skills": guard_report.final_skills,
                "candidate": final_candidate,
                "versions": ExtractionVersions(
                    parser_version=profile.versions.parser_version,
                    matcher_version=profile.versions.matcher_version,
                    ontology_version=profile.versions.ontology_version,
                    guard_version=guard_report.guard_version,
                ),
            }
        )
        # A profile with zero grounded tech skills and clearly non-tech domains is
        # out of this platform's scope: say so plainly instead of forcing 10
        # low-signal tech rows through matching.
        out_of_scope = out_of_scope_reason(final_profile)
        if out_of_scope is not None:
            return CvMatchResult(
                profile=final_profile,
                guard_report=guard_report,
                recommendations=[],
                enrichment_run_id=run_id,
                validation_run_warning=validation_run_warning(run_id),
                status="out_of_scope",
                message=_service_message(fallback_reason, out_of_scope),
            )
        # Low confidence is a WARNING, not a block: it never withholds
        # recommendations, only dampens their score cautiously (cv/confidence.py).
        low_confidence = final_profile.extraction_quality.abstention_reason is not None
        recommendations = await recommend_jobs_for_cv(
            session,
            final_profile,
            run_id=run_id,
            top=top,
            mode=mode,
            alpha=alpha,
            encoder=encoder,
        )
        recommendations = apply_confidence_caution(recommendations, final_profile)
    return CvMatchResult(
        profile=final_profile,
        guard_report=guard_report,
        recommendations=recommendations,
        enrichment_run_id=run_id,
        validation_run_warning=validation_run_warning(run_id),
        status="insufficient_profile_evidence" if low_confidence else "success",
        message=_service_message(fallback_reason, None, low_confidence=low_confidence),
    )


def _service_message(
    fallback_reason: str | None,
    out_of_scope: str | None,
    *,
    low_confidence: bool = False,
) -> str | None:
    parts: list[str] = []
    if fallback_reason is not None:
        parts.append(
            "Richer LLM-based extraction failed, so a simpler deterministic parser "
            "was used instead; experience details may be missing or incomplete, and "
            "any low confidence score reflects that fallback rather than necessarily "
            f"a genuinely weak CV. (LLM failure reason: {fallback_reason})"
        )
    if out_of_scope is not None:
        parts.append(out_of_scope)
    elif low_confidence:
        parts.append(
            "CV extraction confidence is low; recommendations are cautious estimates "
            "and worth a manual review."
        )
    return " ".join(parts) or None


def _merge_candidate_attributes(
    base: CandidateAttributes, extracted: CandidateAttributes
) -> CandidateAttributes:
    """Preserve richer section-based CV facts while accepting final skill-derived facts."""
    domains = _without_unproved_technical_domains(
        sorted({*base.preferred_domains, *extracted.preferred_domains})
    )
    experience = extracted.experience
    if _experience_signal(base.experience) > _experience_signal(extracted.experience):
        experience = base.experience
    return extracted.model_copy(
        update={
            "current_role": extracted.current_role or base.current_role,
            "career_level": base.career_level
            if base.career_level != "unknown"
            else extracted.career_level,
            "education_status": base.education_status or extracted.education_status,
            "experience": experience,
            "preferred_domains": domains,
        }
    )


def _experience_signal(experience: ExperienceSummary) -> tuple[int, int, int]:
    return (
        experience.internship_count,
        experience.professional_experience_count,
        len(experience.evidence),
    )


def _without_unproved_technical_domains(domains: list[str]) -> list[str]:
    technical = {"AI", "Data Science", "Machine Learning", "Software Engineering"}
    business = {"Management", "Customer Service", "Sales", "Insurance", "Communication"}
    values = set(domains)
    if values & business:
        values -= technical
    return sorted(values)
