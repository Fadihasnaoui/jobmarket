"""CV upload, ranked matches, and skill-gap endpoints.

Deliberately reuses the private helpers already built and tested in `cv/workflow.py`
(`_profile_status`, `_profile_output`, `_warnings`, `_combined_message`,
`_recommendation_output`, `_apply_filters_to_profile`, `_dedupe_recommendations`,
`_filter_recommendations`, `_status_from_parse_error`, `_parse_error_message`) instead
of re-deriving equivalent logic — the whole point of this API is to be "as honest as
the CLI", and two independent implementations of "what counts as low confidence" or
"how a recommendation gets its explanation" would drift out of sync over time.

No LLM call and no job-embedding computation happens on any request path here except
the one CV embedding computed once in `upload_cv` — `get_matches`/`get_skill_gap` reuse
that cached vector via `CachedVectorEncoder` rather than re-running the local
sentence-transformers model on every request.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.api.dependencies import (
    get_cv_store,
    get_db_session,
    get_ontology,
    get_upload_encoder,
)
from jobmarket.api.models import CvUploadResponse, MatchesResponse, SkillGapItem, SkillGapResponse
from jobmarket.api.store import CachedVectorEncoder, CvStore, StoredCv
from jobmarket.cv.confidence import apply_confidence_caution
from jobmarket.cv.documents import CvDocumentError, parse_document
from jobmarket.cv.extraction import load_skill_ids
from jobmarket.cv.llm_extraction import extract_cv_profile_with_fallback
from jobmarket.cv.llm_guard import validate_llm_skill_candidates
from jobmarket.cv.matching import (
    DEFAULT_HYBRID_ALPHA,
    RecommendationMode,
    recommend_jobs_for_cv,
    resolve_run_id,
)
from jobmarket.cv.profile import ExtractionVersions, JobRecommendation
from jobmarket.cv.quality_report import detect_contact_signals, enrich_experience_impact_evidence
from jobmarket.cv.scope_guard import out_of_scope_reason
from jobmarket.cv.workflow import (
    CanonicalExtractionStatus,
    RecommendationFilters,
    _apply_filters_to_profile,
    _combined_message,
    _dedupe_recommendations,
    _degraded_message,
    _filter_recommendations,
    _low_confidence_message,
    _parse_error_message,
    _profile_output,
    _profile_status,
    _recommendation_output,
    _status_from_parse_error,
    _warnings,
)
from jobmarket.db.models import JobSkill, Skill
from jobmarket.embeddings.cv import embed_cv_profile
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.skills.ontology import SkillsOntology, load_ontology

router = APIRouter(tags=["cv"])

_ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@router.post("/cv/upload", response_model=CvUploadResponse)
async def upload_cv(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    store: CvStore = Depends(get_cv_store),
    ontology: SkillsOntology = Depends(get_ontology),
    upload_encoder: EmbeddingEncoder | None = Depends(get_upload_encoder),
) -> CvUploadResponse:
    """Parse, extract, guard, and embed one uploaded CV; return a short-lived cv_id.

    The uploaded bytes are written to a temp file only long enough to parse (pypdf/
    python-docx need a path), then deleted immediately — never persisted. The cached
    profile itself has its raw document text stripped before storage (see
    `store.redact_document_text`); only extracted, structured facts are kept.
    """
    filename = file.filename or "cv"
    suffix = Path(filename).suffix.casefold()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported CV file extension: {suffix or '(none)'}. "
                "Use .txt, .pdf, or .docx."
            ),
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # Windows can't unlink a file while its own fd is still open.
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_bytes(raw)
        try:
            document = parse_document(tmp_path)
        except CvDocumentError as exc:
            status = _status_from_parse_error(tmp_path, exc)
            return CvUploadResponse(
                cv_id=None,
                expires_at=store.make_expiry(),
                document_status=status,
                extraction_status=status,
                extraction_method="none",
                extraction_degraded=False,
                out_of_scope=False,
                profile=None,
                warnings=[type(exc).__name__],
                message=_parse_error_message(status, exc),
            )
    finally:
        tmp_path.unlink(missing_ok=True)

    skill_ids = await load_skill_ids(session, ontology)
    extraction_result = extract_cv_profile_with_fallback(
        document, ontology=ontology, skill_ids=skill_ids
    )
    guard_report = validate_llm_skill_candidates(
        document,
        extraction_result.skill_candidates,
        deterministic_skills=extraction_result.profile.skills,
        ontology=ontology,
        skill_ids=skill_ids,
    )
    final_profile = extraction_result.profile.model_copy(
        update={
            "skills": guard_report.final_skills,
            "versions": ExtractionVersions(
                parser_version=extraction_result.profile.versions.parser_version,
                matcher_version=extraction_result.profile.versions.matcher_version,
                ontology_version=extraction_result.profile.versions.ontology_version,
                guard_version=guard_report.guard_version,
            ),
        }
    )
    final_profile = enrich_experience_impact_evidence(final_profile, document.text)
    extraction_status = _profile_status(final_profile, None)
    out_of_scope_message = out_of_scope_reason(final_profile)

    cv_embedding: list[float] | None = None
    if out_of_scope_message is None:
        # Out-of-scope profiles never reach matching (see get_matches), so skip the
        # one real model-inference cost this API ever pays on a request path.
        cv_embedding = embed_cv_profile(final_profile, encoder=upload_encoder)

    # Must happen here, on the freshly parsed document's real text, before `store.put`
    # redacts it below — nothing later in the CV's session can recover this signal.
    contact_signals = detect_contact_signals(document.text)

    item = StoredCv(
        profile=final_profile,
        cv_embedding=cv_embedding,
        extraction_method=extraction_result.extraction_method,
        fallback_reason=extraction_result.fallback_reason,
        out_of_scope_message=out_of_scope_message,
        filename=document.filename,
        created_at=datetime.now(UTC),
        expires_at=store.make_expiry(),
        contact_signals=contact_signals,
    )
    cv_id = store.put(item)

    profile_output = _profile_output(
        final_profile,
        document_status="parsed",
        extraction_status=extraction_status,
        extraction_method=extraction_result.extraction_method,
        fallback_reason=extraction_result.fallback_reason,
    )
    warnings = _warnings(final_profile, extraction_result.fallback_reason, out_of_scope_message)
    message = _upload_message(
        extraction_status, extraction_result.fallback_reason, out_of_scope_message
    )

    return CvUploadResponse(
        cv_id=cv_id,
        expires_at=item.expires_at,
        document_status="parsed",
        extraction_status=extraction_status,
        extraction_method=extraction_result.extraction_method,
        extraction_degraded=extraction_result.fallback_reason is not None,
        extraction_degraded_reason=extraction_result.fallback_reason,
        out_of_scope=out_of_scope_message is not None,
        profile=profile_output,
        warnings=warnings,
        message=message,
    )


@router.get("/cv/{cv_id}/matches", response_model=MatchesResponse)
async def get_matches(
    cv_id: str,
    mode: RecommendationMode = Query("hybrid", description="lexical, semantic, or hybrid"),
    limit: int = Query(10, gt=0, le=100),
    alpha: float = Query(DEFAULT_HYBRID_ALPHA, ge=0.0, le=1.0),
    run_id: int | None = Query(
        None, description="Enrichment run to match against; omit for the latest available run"
    ),
    country: str | None = Query(None),
    contract_type: str | None = Query(None),
    remote: bool | None = Query(None),
    session: AsyncSession = Depends(get_db_session),
    store: CvStore = Depends(get_cv_store),
) -> MatchesResponse:
    started = time.perf_counter()
    try:
        run_id = await resolve_run_id(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stored = _require_stored_cv(store, cv_id)
    extraction_degraded = stored.fallback_reason is not None
    low_confidence = stored.profile.extraction_quality.extraction_confidence_label == "low"

    if stored.out_of_scope_message is not None:
        return MatchesResponse(
            cv_id=cv_id,
            run_id=run_id,
            mode=mode,
            out_of_scope=True,
            extraction_degraded=extraction_degraded,
            low_confidence=low_confidence,
            matches=[],
            warnings=_warnings(stored.profile, stored.fallback_reason, stored.out_of_scope_message),
            message=stored.out_of_scope_message,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )

    filters = RecommendationFilters(
        country=country, contract_type=contract_type, remote_preference=remote
    )
    profile = _apply_filters_to_profile(stored.profile, filters)
    encoder = _cached_encoder(stored, mode)

    try:
        recommendations = await recommend_jobs_for_cv(
            session, profile, run_id=run_id, top=limit, mode=mode, alpha=alpha, encoder=encoder
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    recommendations = _dedupe_recommendations(recommendations)
    recommendations = _filter_recommendations(recommendations, filters)[:limit]
    recommendations = apply_confidence_caution(recommendations, profile)
    outputs = [_recommendation_output(item) for item in recommendations]

    extraction_status = _profile_status(profile, None)
    message = _combined_message(run_id, extraction_status, stored.fallback_reason, None)

    return MatchesResponse(
        cv_id=cv_id,
        run_id=run_id,
        mode=mode,
        out_of_scope=False,
        extraction_degraded=extraction_degraded,
        low_confidence=low_confidence,
        matches=outputs,
        warnings=_warnings(profile, stored.fallback_reason, None),
        message=message,
        elapsed_seconds=round(time.perf_counter() - started, 4),
    )


@router.get("/cv/{cv_id}/skill-gap", response_model=SkillGapResponse)
async def get_skill_gap(
    cv_id: str,
    top: int = Query(20, gt=0, le=100, description="How many top matches to aggregate gaps over"),
    run_id: int | None = Query(
        None, description="Enrichment run to match against; omit for the latest available run"
    ),
    mode: RecommendationMode = Query("hybrid"),
    session: AsyncSession = Depends(get_db_session),
    store: CvStore = Depends(get_cv_store),
) -> SkillGapResponse:
    try:
        run_id = await resolve_run_id(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stored = _require_stored_cv(store, cv_id)

    if stored.out_of_scope_message is not None:
        return SkillGapResponse(
            cv_id=cv_id,
            run_id=run_id,
            top_matches_considered=0,
            out_of_scope=True,
            gaps=[],
            warnings=_warnings(stored.profile, stored.fallback_reason, stored.out_of_scope_message),
            message=stored.out_of_scope_message,
        )

    encoder = _cached_encoder(stored, mode)
    try:
        recommendations = await recommend_jobs_for_cv(
            session, stored.profile, run_id=run_id, top=top, mode=mode, encoder=encoder
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    recommendations = _dedupe_recommendations(recommendations)[:top]
    gaps = await _compute_skill_gaps(session, recommendations, run_id=run_id)

    return SkillGapResponse(
        cv_id=cv_id,
        run_id=run_id,
        top_matches_considered=len(recommendations),
        out_of_scope=False,
        gaps=gaps,
        warnings=[],
        message=None,
    )


async def _compute_skill_gaps(
    session: AsyncSession, recommendations: Sequence[JobRecommendation], *, run_id: int
) -> list[SkillGapItem]:
    """Missing skills across `recommendations`, ranked by real corpus-wide demand.

    Factored out of `get_skill_gap` so `/cv/{id}/improve` can reuse the exact same
    computation (as `SkillGapEntry` inputs to `generate_improved_cv`) instead of
    re-deriving equivalent logic — the same "one tested path" principle the rest of
    this API already follows for profile/recommendation shaping.
    """
    missing_counter: Counter[str] = Counter()
    for rec in recommendations:
        missing_counter.update(rec.missing_skills)
    if not missing_counter:
        return []

    demand = await _skill_demand_for(session, run_id=run_id, canonical_skills=list(missing_counter))
    ontology = load_ontology()
    gaps = [
        SkillGapItem(
            canonical_skill=skill,
            category=(entry.category if (entry := ontology.get(skill)) is not None else None),
            missing_in_top_matches=missing_counter[skill],
            overall_job_demand=demand.get(skill, 0),
        )
        for skill in missing_counter
    ]
    gaps.sort(
        key=lambda item: (
            -item.overall_job_demand,
            -item.missing_in_top_matches,
            item.canonical_skill,
        )
    )
    return gaps


def _upload_message(
    status: CanonicalExtractionStatus,
    fallback_reason: str | None,
    out_of_scope_message: str | None,
) -> str | None:
    """Upload has no run_id yet, so unlike `_combined_message` it never attaches the
    per-run validation warning — that's added once a run is actually chosen, in
    `get_matches`/`get_skill_gap`.
    """
    parts: list[str | None] = [_degraded_message(fallback_reason)]
    parts.append(
        out_of_scope_message
        if out_of_scope_message is not None
        else _low_confidence_message(status)
    )
    combined = " ".join(part for part in parts if part)
    return combined or None


def _cached_encoder(stored: StoredCv, mode: RecommendationMode) -> CachedVectorEncoder | None:
    if mode == "lexical" or stored.cv_embedding is None:
        return None
    return CachedVectorEncoder(stored.cv_embedding)


def _require_stored_cv(store: CvStore, cv_id: str) -> StoredCv:
    stored = store.get(cv_id)
    if stored is None:
        raise HTTPException(status_code=404, detail="Unknown or expired cv_id.")
    return stored


async def _skill_demand_for(
    session: AsyncSession, *, run_id: int, canonical_skills: list[str]
) -> dict[str, int]:
    if not canonical_skills:
        return {}
    rows = await session.execute(
        select(Skill.canonical, func.count(func.distinct(JobSkill.job_id)))
        .select_from(JobSkill)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id, Skill.canonical.in_(canonical_skills))
        .group_by(Skill.canonical)
    )
    return {canonical: int(count) for canonical, count in rows.all()}


__all__ = ["router"]
