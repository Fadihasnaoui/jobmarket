"""Data-driven CV improvement endpoint: reworded CV + downloadable .docx.

Reuses the same tested helpers as `cv_routes.py` (`_require_stored_cv`,
`_cached_encoder`, `_compute_skill_gaps`, `_dedupe_recommendations`,
`_recommendation_output`, `_warnings`) rather than re-deriving equivalent logic —
matching/gaps are computed identically to `/cv/{id}/matches` and `/cv/{id}/skill-gap`.

The one real LLM call (`generate_improved_cv`) is cached per `cv_id` in `CvStore` so
repeated calls — e.g. regenerating the .docx with a different name/contact — don't
re-trigger it. Name/contact never touch that cache: `render_improved_cv_docx` takes
them directly per-request and they are discarded the moment the response is built.
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
from jobmarket.api.models import ImproveCvRequest, ImproveCvResponse
from jobmarket.api.store import CvStore
from jobmarket.cv.docx_export import render_improved_cv_docx
from jobmarket.cv.improver import CvImproverError, SkillGapEntry, generate_improved_cv
from jobmarket.cv.matching import RecommendationMode, recommend_jobs_for_cv, resolve_run_id
from jobmarket.cv.workflow import _dedupe_recommendations, _recommendation_output, _warnings

router = APIRouter(tags=["cv"])

_MATCHED_JOBS_CONSIDERED = 20


@router.post("/cv/{cv_id}/improve", response_model=ImproveCvResponse)
async def improve_cv(
    cv_id: str,
    body: ImproveCvRequest = Body(default_factory=ImproveCvRequest),
    run_id: int | None = Query(
        None, description="Enrichment run to match against; omit for the latest available run"
    ),
    mode: RecommendationMode = Query("hybrid"),
    session: AsyncSession = Depends(get_db_session),
    store: CvStore = Depends(get_cv_store),
) -> ImproveCvResponse:
    try:
        run_id = await resolve_run_id(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    stored = _require_stored_cv(store, cv_id)
    extraction_degraded = stored.fallback_reason is not None
    low_confidence = stored.profile.extraction_quality.extraction_confidence_label == "low"

    if stored.out_of_scope_message is not None:
        return ImproveCvResponse(
            cv_id=cv_id,
            out_of_scope=True,
            extraction_degraded=extraction_degraded,
            low_confidence=low_confidence,
            warnings=_warnings(stored.profile, stored.fallback_reason, stored.out_of_scope_message),
            message=stored.out_of_scope_message,
        )

    result = stored.improved_result
    if result is None:
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

        try:
            result = generate_improved_cv(stored.profile, skill_gap, matched_jobs)
        except CvImproverError as exc:
            raise HTTPException(
                status_code=502, detail=f"Could not generate an improved CV: {exc}"
            ) from exc
        store.set_improved_result(cv_id, result)

    docx_bytes = render_improved_cv_docx(
        result,
        stored.profile,
        candidate_name=body.candidate_name,
        contact_info=body.contact_info,
    )

    return ImproveCvResponse(
        cv_id=cv_id,
        out_of_scope=False,
        extraction_degraded=extraction_degraded,
        low_confidence=low_confidence,
        improved_summary=result.improved_summary,
        improved_experiences=result.improved_experiences,
        skills_section=result.skills_section,
        recommended_skills_to_develop=result.recommended_skills_to_develop,
        changes_explanation=result.changes_explanation,
        source_degraded=result.source_degraded,
        source_degraded_reason=result.source_degraded_reason,
        docx_base64=base64.b64encode(docx_bytes).decode("ascii"),
        warnings=_warnings(stored.profile, stored.fallback_reason, None),
        message=None,
    )


__all__ = ["router"]
