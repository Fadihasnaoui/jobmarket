"""End-to-end CV upload to job recommendation workflow."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from jobmarket.cv.confidence import apply_confidence_caution
from jobmarket.cv.documents import CvDocumentError, EmptyDocumentError, parse_document
from jobmarket.cv.extraction import load_skill_ids
from jobmarket.cv.llm_extraction import CvExtractionResult, extract_cv_profile_with_fallback
from jobmarket.cv.llm_guard import validate_llm_skill_candidates
from jobmarket.cv.matching import (
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RECOMMENDATION_RUN_ID,
    RecommendationMode,
    recommend_jobs_for_cv,
    validation_run_warning,
)
from jobmarket.cv.profile import (
    CvProfile,
    CvSkill,
    EvidenceSpan,
    ExperienceSummary,
    ExtractionVersions,
    JobRecommendation,
    ParsedDocument,
)
from jobmarket.cv.scope_guard import out_of_scope_reason
from jobmarket.db.session import async_session_factory
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import SkillsOntology, load_ontology

logger = structlog.get_logger(__name__)

DocumentStatus = Literal["parsed", "parse_error", "ocr_required"]
CanonicalExtractionStatus = Literal[
    "success",
    "insufficient_profile_evidence",
    "ocr_required",
    "parse_error",
]


class RecommendationFilters(BaseModel):
    """Optional user-facing filters and preferences for matching."""

    model_config = ConfigDict(frozen=True)

    country: str | None = None
    city: str | None = None
    contract_type: str | None = None
    remote_preference: bool | None = None


class ConfirmedProfileInput(BaseModel):
    """User-reviewed profile facts allowed to drive matching."""

    model_config = ConfigDict(frozen=True)

    canonical_skills: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    career_level: str = "unknown"
    education_status: str | None = None
    total_years_experience: float | None = None
    internship_count: int = 0
    professional_experience_count: int = 0


class CvRecommendationRequest(BaseModel):
    """Typed request for the complete CV recommendation workflow."""

    model_config = ConfigDict(frozen=True)

    cv_path: str
    limit: int = 10
    run_id: int = DEFAULT_RECOMMENDATION_RUN_ID
    mode: RecommendationMode = "lexical"
    alpha: float = DEFAULT_HYBRID_ALPHA
    filters: RecommendationFilters = Field(default_factory=RecommendationFilters)
    confirmed_profile: ConfirmedProfileInput | None = None


class ExtractedProfileOutput(BaseModel):
    """Privacy-safe profile facts returned for user review."""

    model_config = ConfigDict(frozen=True)

    filename: str
    mime_type: str | None = None
    document_status: DocumentStatus
    canonical_extraction_status: CanonicalExtractionStatus
    ocr_required: bool
    extraction_confidence_score: float | None = None
    extraction_confidence_label: str | None = None
    text_quality: float | None = None
    section_coverage: float | None = None
    education: dict[str, object] | None = None
    experiences: list[dict[str, object]] = Field(default_factory=list)
    internships: list[dict[str, object]] = Field(default_factory=list)
    canonical_skills: list[str] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    career_level: str = "unknown"
    extraction_warnings: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None
    extraction_method: str = "deterministic"
    extraction_degraded: bool = False
    extraction_degraded_reason: str | None = None


class RecommendationOutput(BaseModel):
    """User-facing explainable recommendation shape."""

    model_config = ConfigDict(frozen=True)

    job_id: int
    title: str
    company: str
    location: str | None = None
    contract_type: str | None = None
    final_score: float
    lexical_score: float | None = None
    semantic_score: float | None = None
    retrieval_source: str = "lexical"
    skill_score: float
    role_domain_score: float
    career_level_compatibility: float
    matched_skills: list[str]
    missing_important_skills: list[str]
    explanation: str
    source_url: str | None = None
    posted_at: datetime | None = None


class CvRecommendationResponse(BaseModel):
    """Complete privacy-safe workflow response."""

    model_config = ConfigDict(frozen=True)

    document_status: DocumentStatus
    canonical_extraction_status: CanonicalExtractionStatus
    ocr_required: bool
    profile: ExtractedProfileOutput | None
    recommendations: list[RecommendationOutput]
    message: str | None = None
    warnings: list[str] = Field(default_factory=list)
    enrichment_run_id: int = DEFAULT_RECOMMENDATION_RUN_ID
    elapsed_seconds: float = 0.0
    diagnostics: dict[str, object] = Field(default_factory=dict)


async def run_cv_recommendation_workflow(
    request: CvRecommendationRequest,
    *,
    encoder: EmbeddingEncoder | None = None,
) -> CvRecommendationResponse:
    """Run CV parsing, profile extraction, quality gating, and job recommendation."""
    if request.limit <= 0:
        raise ValueError("limit must be positive")
    started = time.perf_counter()
    stage_times: dict[str, float] = {}
    cv_path = Path(request.cv_path)
    log = logger.bind(
        workflow="cv_recommendation",
        filename=cv_path.name,
        run_id=request.run_id,
        mode=request.mode,
        limit=request.limit,
    )

    parse_started = time.perf_counter()
    try:
        document = parse_document(cv_path)
    except CvDocumentError as exc:
        status = _status_from_parse_error(cv_path, exc)
        elapsed = round(time.perf_counter() - started, 4)
        _log_info(
            log,
            "cv_document_extraction_failed",
            document_status=status,
            parser_status=status,
            error_type=type(exc).__name__,
            elapsed_seconds=elapsed,
        )
        return CvRecommendationResponse(
            document_status=status,
            canonical_extraction_status=status,
            ocr_required=status == "ocr_required",
            profile=ExtractedProfileOutput(
                filename=cv_path.name,
                mime_type=None,
                document_status=status,
                canonical_extraction_status=status,
                ocr_required=status == "ocr_required",
                extraction_warnings=[type(exc).__name__],
            ),
            recommendations=[],
            message=_parse_error_message(status, exc),
            warnings=[type(exc).__name__],
            enrichment_run_id=request.run_id,
            elapsed_seconds=elapsed,
            diagnostics={"fallback_path_used": "quality_gate_parse_error"},
        )
    stage_times["document_extraction_seconds"] = round(time.perf_counter() - parse_started, 4)
    _log_info(
        log,
        "cv_document_extracted",
        document_status="parsed",
        mime_type=document.mime_type,
        page_count=document.page_count,
        file_size=document.file_size,
    )

    ontology = load_ontology()
    async with async_session_factory() as session:
        extraction_started = time.perf_counter()
        skill_ids = await load_skill_ids(session, ontology)
        extraction_result: CvExtractionResult = extract_cv_profile_with_fallback(
            document, ontology=ontology, skill_ids=skill_ids
        )
        profile = extraction_result.profile
        skill_candidates = extraction_result.skill_candidates
        extraction_method = extraction_result.extraction_method
        fallback_reason = extraction_result.fallback_reason
        if request.confirmed_profile is not None:
            profile = _apply_confirmed_profile(
                profile,
                request.confirmed_profile,
                ontology=ontology,
                skill_ids=skill_ids,
            )
            skill_candidates = []
        profile = _apply_filters_to_profile(profile, request.filters)
        guard_report = validate_llm_skill_candidates(
            document,
            skill_candidates,
            deterministic_skills=profile.skills,
            ontology=ontology,
            skill_ids=skill_ids,
        )
        final_profile = profile.model_copy(
            update={
                "skills": guard_report.final_skills,
                "versions": ExtractionVersions(
                    parser_version=profile.versions.parser_version,
                    matcher_version=profile.versions.matcher_version,
                    ontology_version=profile.versions.ontology_version,
                    guard_version=guard_report.guard_version,
                ),
            }
        )
        extraction_status = _profile_status(final_profile, request.confirmed_profile)
        profile_output = _profile_output(
            final_profile,
            document_status="parsed",
            extraction_status=extraction_status,
            extraction_method=extraction_method,
            fallback_reason=fallback_reason,
        )
        stage_times["profile_extraction_seconds"] = round(
            time.perf_counter() - extraction_started,
            4,
        )
        _log_info(
            log,
            "cv_profile_extracted",
            parser_status=extraction_status,
            confidence=final_profile.extraction_quality.extraction_confidence_score,
            confidence_label=final_profile.extraction_quality.extraction_confidence_label,
            skill_count=len(final_profile.skills),
            role_count=len(final_profile.inferred_roles),
            domain_count=len(final_profile.candidate.preferred_domains),
            extraction_method=extraction_method,
            extraction_degraded=fallback_reason is not None,
        )

        # A profile with zero grounded tech skills and clearly non-tech domains (e.g.
        # a Sales/Customer-Service CV) is out of this platform's scope: say so plainly
        # instead of running matching down to 10 near-random low-signal rows.
        out_of_scope = out_of_scope_reason(final_profile)
        if out_of_scope is not None:
            elapsed = round(time.perf_counter() - started, 4)
            _log_info(
                log,
                "cv_recommendation_out_of_scope",
                parser_status=extraction_status,
                extraction_method=extraction_method,
                elapsed_seconds=elapsed,
            )
            return CvRecommendationResponse(
                document_status="parsed",
                canonical_extraction_status=extraction_status,
                ocr_required=False,
                profile=profile_output,
                recommendations=[],
                message=_combined_message(
                    request.run_id, extraction_status, fallback_reason, out_of_scope
                ),
                warnings=_warnings(final_profile, fallback_reason, out_of_scope),
                enrichment_run_id=request.run_id,
                elapsed_seconds=elapsed,
                diagnostics={
                    "stage_times": stage_times,
                    "filters_applied": request.filters.model_dump(),
                    "recommendation_candidate_count": 0,
                    "fallback_path_used": "out_of_scope",
                    "extraction_method": extraction_method,
                    "extraction_degraded": fallback_reason is not None,
                },
            )

        # Low confidence (e.g. a student with no formal experience) is a WARNING, not a
        # block: it never withholds recommendations, only dampens their score cautiously
        # (see cv/confidence.py) and surfaces the reason in `warnings`/`message`.
        matching_started = time.perf_counter()
        recommendations = await recommend_jobs_for_cv(
            session,
            final_profile,
            run_id=request.run_id,
            top=request.limit,
            mode=request.mode,
            alpha=request.alpha,
            encoder=encoder,
        )
        recommendations = _dedupe_recommendations(recommendations)
        recommendations = _filter_recommendations(recommendations, request.filters)[: request.limit]
        recommendations = apply_confidence_caution(recommendations, final_profile)
        stage_times["recommendation_seconds"] = round(time.perf_counter() - matching_started, 4)

    elapsed = round(time.perf_counter() - started, 4)
    outputs = [_recommendation_output(item) for item in recommendations]
    caution_applied = extraction_status == "insufficient_profile_evidence"
    _log_info(
        log,
        "cv_recommendations_completed",
        parser_status=extraction_status,
        recommendation_candidate_count=len(recommendations),
        final_recommendation_count=len(outputs),
        filters_applied=request.filters.model_dump(),
        fallback_path_used="none",
        extraction_method=extraction_method,
        low_confidence_caution_applied=caution_applied,
        elapsed_seconds=elapsed,
    )
    return CvRecommendationResponse(
        document_status="parsed",
        canonical_extraction_status=extraction_status,
        ocr_required=False,
        profile=profile_output,
        recommendations=outputs,
        message=_combined_message(request.run_id, extraction_status, fallback_reason, None),
        warnings=_warnings(final_profile, fallback_reason, None),
        enrichment_run_id=request.run_id,
        elapsed_seconds=elapsed,
        diagnostics={
            "stage_times": stage_times,
            "filters_applied": request.filters.model_dump(),
            "recommendation_candidate_count": len(recommendations),
            "fallback_path_used": "none",
            "extraction_method": extraction_method,
            "extraction_degraded": fallback_reason is not None,
            "low_confidence_caution_applied": caution_applied,
        },
    )


def workflow_response_to_json(response: CvRecommendationResponse) -> str:
    """Serialize a workflow response without raw CV text or embeddings."""
    return json.dumps(response.model_dump(mode="json"), indent=2, sort_keys=True)


def format_workflow_response(response: CvRecommendationResponse) -> str:
    """Render a concise human-readable workflow response."""
    lines = [
        "CV recommendation workflow",
        "==========================",
        f"document_status:        {response.document_status}",
        f"extraction_status:      {response.canonical_extraction_status}",
        f"ocr_required:           {response.ocr_required}",
        f"enrichment_run_id:      {response.enrichment_run_id}",
    ]
    if response.message:
        lines.append(f"message:                {response.message}")
    if response.profile is not None:
        profile = response.profile
        lines.extend(
            [
                "",
                "Extracted profile",
                "-----------------",
                f"filename:               {profile.filename}",
                f"extraction_method:      {profile.extraction_method}"
                + ("  *** DEGRADED — see below ***" if profile.extraction_degraded else ""),
                f"confidence:             {profile.extraction_confidence_label} "
                f"({profile.extraction_confidence_score})",
                f"career_level:           {profile.career_level}",
                f"skills:                 {', '.join(profile.canonical_skills) or 'none'}",
                f"domains:                {', '.join(profile.domains) or 'none'}",
                f"roles:                  {', '.join(profile.role_families) or 'none'}",
                f"internships:            {len(profile.internships)}",
                f"experiences:            {len(profile.experiences)}",
            ]
        )
        if profile.extraction_degraded:
            lines.append(
                f"extraction_degraded_reason: {profile.extraction_degraded_reason}"
            )
    if response.warnings:
        lines.extend(["", "Warnings", "--------"])
        lines.extend(f"- {warning}" for warning in response.warnings)
    lines.extend(["", "Recommendations", "---------------"])
    if not response.recommendations:
        lines.append("(none)")
    for index, rec in enumerate(response.recommendations, start=1):
        lines.extend(
            [
                f"{index}. job {rec.job_id}: {rec.title} ({rec.company})",
                f"   score={rec.final_score:.2f} "
                f"lexical={_format_optional_score(rec.lexical_score)} "
                f"semantic={_format_optional_score(rec.semantic_score)} "
                f"retrieval_source={rec.retrieval_source}",
                f"   skill={rec.skill_score:.2f} "
                f"role_domain={rec.role_domain_score:.2f} "
                f"career={rec.career_level_compatibility:.2f}",
                (
                    f"   location={rec.location or 'unknown'} "
                    f"contract={rec.contract_type or 'unknown'}"
                ),
                f"   matched: {', '.join(rec.matched_skills) or 'none'}",
                f"   missing important: {', '.join(rec.missing_important_skills) or 'none'}",
                f"   source: {rec.source_url or 'n/a'}",
                f"   {rec.explanation}",
            ]
        )
    return "\n".join(lines)


def _format_optional_score(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/a"


def _log_info(log: Any, event: str, **values: object) -> None:
    try:
        log.info(event, **values)
    except ValueError:
        # Test harnesses may close captured stdout after Typer configures structlog.
        return


def _status_from_parse_error(path: Path, exc: CvDocumentError) -> DocumentStatus:
    if isinstance(exc, EmptyDocumentError) and path.suffix.casefold() == ".pdf":
        return "ocr_required"
    return "parse_error"


def _parse_error_message(status: DocumentStatus, exc: CvDocumentError) -> str:
    if status == "ocr_required":
        return "The CV appears to require OCR before recommendations can be generated."
    return f"The CV could not be parsed safely ({type(exc).__name__})."


def _profile_status(
    profile: CvProfile,
    confirmed_profile: ConfirmedProfileInput | None,
) -> CanonicalExtractionStatus:
    if confirmed_profile is not None and profile.skills:
        return "success"
    if profile.extraction_quality.abstention_reason is not None:
        return "insufficient_profile_evidence"
    if not profile.skills:
        return "insufficient_profile_evidence"
    return "success"


def _combined_message(
    run_id: int,
    status: CanonicalExtractionStatus,
    fallback_reason: str | None,
    out_of_scope: str | None,
) -> str | None:
    parts: list[str | None] = [_degraded_message(fallback_reason)]
    parts.append(out_of_scope if out_of_scope is not None else _low_confidence_message(status))
    parts.append(validation_run_warning(run_id))
    combined = " ".join(part for part in parts if part)
    return combined or None


def _degraded_message(fallback_reason: str | None) -> str | None:
    if fallback_reason is None:
        return None
    return (
        "Richer LLM-based extraction failed, so a simpler deterministic parser was "
        "used instead; experience details may be missing or incomplete, and any low "
        "confidence score below reflects that fallback rather than necessarily a "
        f"genuinely weak CV. (LLM failure reason: {fallback_reason})"
    )


def _low_confidence_message(status: CanonicalExtractionStatus) -> str | None:
    if status == "insufficient_profile_evidence":
        return (
            "CV extraction confidence is low; recommendations below are cautious "
            "estimates and worth a manual review."
        )
    return None


def _apply_filters_to_profile(profile: CvProfile, filters: RecommendationFilters) -> CvProfile:
    candidate = profile.candidate.model_copy(
        update={
            "preferred_country": filters.country or profile.candidate.preferred_country,
            "preferred_city": filters.city or profile.candidate.preferred_city,
            "remote_preference": (
                filters.remote_preference
                if filters.remote_preference is not None
                else profile.candidate.remote_preference
            ),
        }
    )
    return profile.model_copy(update={"candidate": candidate})


def _apply_confirmed_profile(
    profile: CvProfile,
    confirmed: ConfirmedProfileInput,
    *,
    ontology: SkillsOntology,
    skill_ids: dict[str, int],
) -> CvProfile:
    skills = [
        _confirmed_skill(raw_skill, profile.document, ontology=ontology, skill_ids=skill_ids)
        for raw_skill in confirmed.canonical_skills
    ]
    candidate = profile.candidate.model_copy(
        update={
            "career_level": confirmed.career_level,
            "education_status": confirmed.education_status,
            "preferred_domains": sorted(set(confirmed.domains)),
            "experience": ExperienceSummary(
                internship_count=confirmed.internship_count,
                professional_experience_count=confirmed.professional_experience_count,
                total_years=confirmed.total_years_experience,
                total_months=(
                    int(round(confirmed.total_years_experience * 12))
                    if confirmed.total_years_experience is not None
                    else None
                ),
                evidence=["user_confirmed_profile"],
            ),
        }
    )
    return profile.model_copy(update={"skills": skills, "candidate": candidate})


def _confirmed_skill(
    raw_skill: str,
    document: ParsedDocument,
    *,
    ontology: SkillsOntology,
    skill_ids: dict[str, int],
) -> CvSkill:
    canonical = ontology.resolve(raw_skill) or raw_skill
    skill_id = skill_ids.get(canonical)
    if skill_id is None:
        raise ValueError(f"Confirmed skill is not in the ontology: {raw_skill}")
    matcher_versions = get_matcher_versions(ontology)
    versions = ExtractionVersions(
        parser_version=document.parser_version,
        matcher_version=matcher_versions.matcher_version,
        ontology_version=matcher_versions.ontology_version,
    )
    return CvSkill(
        canonical_skill=canonical,
        ontology_skill_id=skill_id,
        matched_alias=raw_skill,
        extraction_method="deterministic",
        evidence=EvidenceSpan(
            evidence_text="",
            page=None,
            document_start=0,
            document_end=0,
            page_start=None,
            page_end=None,
        ),
        matcher_version=versions.matcher_version,
        ontology_version=versions.ontology_version,
        validation_status="deterministic",
        confidence=1.0,
    )


def _profile_output(
    profile: CvProfile,
    *,
    document_status: DocumentStatus,
    extraction_status: CanonicalExtractionStatus,
    extraction_method: str = "deterministic",
    fallback_reason: str | None = None,
) -> ExtractedProfileOutput:
    education = (
        profile.education_detail.model_dump(mode="json")
        if profile.education_detail
        else None
    )
    experiences = [entry.model_dump(mode="json") for entry in profile.experience_entries]
    internships = [entry for entry in experiences if entry.get("entry_type") == "internship"]
    quality = profile.extraction_quality
    return ExtractedProfileOutput(
        filename=profile.document.filename,
        mime_type=profile.document.mime_type,
        document_status=document_status,
        canonical_extraction_status=extraction_status,
        ocr_required=extraction_status == "ocr_required",
        extraction_confidence_score=quality.extraction_confidence_score,
        extraction_confidence_label=quality.extraction_confidence_label,
        text_quality=quality.text_quality,
        section_coverage=quality.section_coverage,
        education=education,
        experiences=experiences,
        internships=internships,
        canonical_skills=sorted({skill.canonical_skill for skill in profile.skills}),
        role_families=sorted({role.value for role in profile.inferred_roles}),
        domains=sorted(set(profile.candidate.preferred_domains)),
        career_level=profile.candidate.career_level,
        extraction_warnings=_warnings(profile, fallback_reason, None),
        abstention_reason=quality.abstention_reason,
        extraction_method=extraction_method,
        extraction_degraded=fallback_reason is not None,
        extraction_degraded_reason=fallback_reason,
    )


def _warnings(
    profile: CvProfile,
    fallback_reason: str | None = None,
    out_of_scope: str | None = None,
) -> list[str]:
    values = [warning.code for warning in profile.document.warnings]
    values.extend(profile.extraction_quality.warnings)
    if profile.extraction_quality.abstention_reason:
        values.append(profile.extraction_quality.abstention_reason)
    if fallback_reason is not None:
        values.append("llm_extraction_degraded")
    if out_of_scope is not None:
        values.append("out_of_scope_non_tech_profile")
    return sorted(set(values))


def _dedupe_recommendations(items: list[JobRecommendation]) -> list[JobRecommendation]:
    by_id: dict[int, JobRecommendation] = {}
    for item in sorted(items, key=lambda rec: (-rec.final_score, rec.job_id)):
        by_id.setdefault(item.job_id, item)
    return sorted(by_id.values(), key=lambda rec: (-rec.final_score, rec.job_id))


def _filter_recommendations(
    items: list[JobRecommendation],
    filters: RecommendationFilters,
) -> list[JobRecommendation]:
    filtered = items
    if filters.contract_type:
        wanted = filters.contract_type.casefold()
        filtered = [
            item
            for item in filtered
            if item.contract_type is not None and item.contract_type.casefold() == wanted
        ]
    return filtered


def _recommendation_output(item: JobRecommendation) -> RecommendationOutput:
    components = item.score_components
    important_missing = sorted(
        item.missing_skills,
        key=lambda skill: (0 if skill in set(item.specialized_matched_skills) else 1, skill),
    )[:5]
    return RecommendationOutput(
        job_id=item.job_id,
        title=item.title,
        company=item.company,
        location=item.location,
        contract_type=item.contract_type,
        final_score=item.final_score,
        lexical_score=components.lexical_score if components is not None else None,
        semantic_score=item.semantic_score,
        retrieval_source=item.retrieval_source,
        skill_score=components.skill_score if components is not None else item.coverage_score,
        role_domain_score=components.domain_score if components is not None else 0.0,
        career_level_compatibility=(
            components.career_level_score if components is not None else 0.0
        ),
        matched_skills=item.matched_skills,
        missing_important_skills=important_missing,
        explanation=_deterministic_explanation(item),
        source_url=item.source_url,
        posted_at=item.posted_at,
    )


def _deterministic_explanation(item: JobRecommendation) -> str:
    matched = ", ".join(item.matched_skills[:5]) or "no explicit skills"
    parts = [f"Recommended because the CV and job share {matched}."]
    if item.score_components is not None:
        if item.score_components.career_level_score >= 0.75:
            parts.append("The candidate's career level is compatible.")
        if item.score_components.domain_score >= 0.75:
            parts.append("The role/domain evidence is compatible.")
    if item.missing_skills:
        parts.append(f"{item.missing_skills[0]} is listed as a missing important skill.")
    if item.penalties:
        penalty_codes = ", ".join(p.code for p in item.penalties)
        parts.append(f"Compatibility penalties were applied: {penalty_codes}.")
    return " ".join(parts)


__all__ = [
    "ConfirmedProfileInput",
    "CvRecommendationRequest",
    "CvRecommendationResponse",
    "ExtractedProfileOutput",
    "RecommendationFilters",
    "RecommendationOutput",
    "format_workflow_response",
    "run_cv_recommendation_workflow",
    "workflow_response_to_json",
]
