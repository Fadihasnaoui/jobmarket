"""CV Quality Report endpoint: scored diagnostic + two-column .docx download.

Mirrors `improve_routes.py`'s structure and reuses the same tested helpers from
`cv_routes.py` (`_require_stored_cv`, `_cached_encoder`, `_compute_skill_gaps`) —
matches/gaps are computed identically to `/cv/{id}/matches` and `/cv/{id}/improve`.

Unlike `/cv/{id}/improve`, the report itself is deliberately NOT cached in `CvStore`:
`generate_cv_quality_report`'s "missing contact" check depends on `candidate_name`/
`contact_info`, which can legitimately change between requests for the same cv_id
(e.g. a user fills in contact info and re-checks) — caching the report would freeze
that check's result to whatever was passed on the first call, which is exactly the
kind of stale-but-plausible-looking result this whole feature exists to avoid. The
`ImprovedCvResult` needed for the two-column .docx has no such dependency, so it
reuses/populates the same `improved_result` cache `/cv/{id}/improve` already uses.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.api.cv_routes import (
    _cached_encoder,
    _compute_skill_gaps,
    _require_stored_cv,
)
from jobmarket.api.dependencies import get_cv_store, get_db_session
from jobmarket.api.models import QualityReportRequest, QualityReportResponse
from jobmarket.api.store import CvStore
from jobmarket.cv.docx_export import render_two_column_cv_docx
from jobmarket.cv.improver import CvImproverError, SkillGapEntry, generate_improved_cv
from jobmarket.cv.matching import RecommendationMode, recommend_jobs_for_cv, resolve_run_id
from jobmarket.cv.quality_report import generate_cv_quality_report
from jobmarket.cv.workflow import _dedupe_recommendations, _recommendation_output, _warnings

router = APIRouter(tags=["cv"])

_MATCHED_JOBS_CONSIDERED = 20


@router.post("/cv/{cv_id}/quality-report", response_model=QualityReportResponse)
async def get_quality_report(
    cv_id: str,
    body: QualityReportRequest = Body(default_factory=QualityReportRequest),
    run_id: int | None = Query(
        None, description="Enrichment run to match against; omit for the latest available run"
    ),
    mode: RecommendationMode = Query("hybrid"),
    session: AsyncSession = Depends(get_db_session),
    store: CvStore = Depends(get_cv_store),
) -> QualityReportResponse:
    try:
        run_id = await resolve_run_id(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stored = _require_stored_cv(store, cv_id)
    extraction_degraded = stored.fallback_reason is not None
    low_confidence = stored.profile.extraction_quality.extraction_confidence_label == "low"

    if stored.out_of_scope_message is not None:
        return QualityReportResponse(
            cv_id=cv_id,
            out_of_scope=True,
            extraction_degraded=extraction_degraded,
            low_confidence=low_confidence,
            warnings=_warnings(stored.profile, stored.fallback_reason, stored.out_of_scope_message),
            message=stored.out_of_scope_message,
        )

    encoder = _cached_encoder(stored, mode)
    try:
        recommendations = await recommend_jobs_for_cv(
            session,
            stored.profile,
            run_id=run_id,
            top=_MATCHED_JOBS_CONSIDERED,
            mode=mode,
            encoder=encoder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    recommendations = _dedupe_recommendations(recommendations)[:_MATCHED_JOBS_CONSIDERED]
    matched_jobs = [_recommendation_output(item) for item in recommendations]
    gap_items = await _compute_skill_gaps(session, recommendations, run_id=run_id)
    skill_gap = [SkillGapEntry(**item.model_dump()) for item in gap_items]

    report = generate_cv_quality_report(
        stored.profile,
        skill_gap,
        matched_jobs,
        candidate_name=body.candidate_name,
        contact_info=body.contact_info,
        contact_signals=stored.contact_signals,
    )

    improved_result = stored.improved_result
    if improved_result is None:
        try:
            improved_result = generate_improved_cv(stored.profile, skill_gap, matched_jobs)
        except CvImproverError as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not generate the downloadable CV: {exc}"
            ) from exc
        store.set_improved_result(cv_id, improved_result)

    docx_bytes = render_two_column_cv_docx(
        improved_result,
        stored.profile,
        candidate_name=body.candidate_name,
        candidate_title=body.candidate_title,
        contact_info=body.contact_info,
    )

    return QualityReportResponse(
        cv_id=cv_id,
        out_of_scope=False,
        extraction_degraded=extraction_degraded,
        low_confidence=low_confidence,
        overall_score=report.overall_score,
        checks=report.checks,
        spelling_issues=report.spelling_issues,
        unquantified_bullets=report.unquantified_bullets,
        recommended_skills_to_develop=report.recommended_skills_to_develop,
        dropped_spelling_suggestions=report.dropped_spelling_suggestions,
        source_degraded=report.source_degraded,
        source_degraded_reason=report.source_degraded_reason,
        docx_base64=base64.b64encode(docx_bytes).decode("ascii"),
        warnings=_warnings(stored.profile, stored.fallback_reason, None),
        message=None,
    )


__all__ = ["router"]
