"""Corpus overview and skill-demand endpoints.

Reuses `reporting/stats.py::collect_stats` for the raw ingestion totals (jobs,
companies, per-source breakdown) rather than re-querying that, then adds
enrichment/embedding coverage and skill-demand queries specific to this API.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.api.dependencies import get_db_session
from jobmarket.api.models import (
    SkillDemandItem,
    SourceStatsOut,
    StatsOverviewResponse,
    StatsSkillsResponse,
)
from jobmarket.cv.matching import resolve_run_id
from jobmarket.db.models import EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.reporting.stats import collect_stats
from jobmarket.skills.ontology import load_ontology

router = APIRouter(tags=["stats"])

# Verified against the live DB: these must be literal single-backslash `\m`/`\M`
# (Postgres word-boundary escapes) since the pattern is sent as a bind parameter, not
# embedded in the SQL text — a double-backslash version (which the existing
# `embeddings/repository.py::_career_level_pattern` uses) silently never matches
# anything. Not fixing that pre-existing function here (out of scope for this task);
# flagged separately.
_SENIORITY_PATTERNS: dict[str, str] = {
    "junior": r"\mjunior\M|\mdebutant\M",
    "senior": r"\msenior\M|\msr\M|\mconfirme\M|\mexperimente\M",
    "lead": r"\mlead\M|tech lead|leader technique",
    "manager": r"\mmanager\M|responsable|head of|chef de projet",
    "internship": r"\mstage\M|stagiaire|internship|intern",
    "apprenticeship": r"alternance|apprenti|apprentice",
}


@router.get("/stats/overview", response_model=StatsOverviewResponse)
async def stats_overview(
    run_id: int | None = Query(
        None, description="Reference enrichment run; omit for the latest available run"
    ),
    session: AsyncSession = Depends(get_db_session),
) -> StatsOverviewResponse:
    base = await collect_stats()

    try:
        run_id = await resolve_run_id(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run = await session.get(EnrichmentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Enrichment run {run_id} not found.")

    embedded_jobs = await session.scalar(
        text("SELECT count(*) FROM jobs WHERE embedding IS NOT NULL")
    )
    distinct_skills_matched = await session.scalar(
        select(func.count(func.distinct(JobSkill.skill_id))).where(JobSkill.run_id == run_id)
    )
    skill_coverage_pct = (run.jobs_with_skills / run.jobs_total * 100) if run.jobs_total else 0.0
    embedding_coverage_pct = (int(embedded_jobs or 0) / base.jobs * 100) if base.jobs else 0.0

    return StatsOverviewResponse(
        raw_jobs=base.raw_jobs,
        jobs=base.jobs,
        companies=base.companies,
        job_sources=base.job_sources,
        per_source=[
            SourceStatsOut(
                source=item.source,
                raw_total=item.raw_total,
                parsed=item.parsed,
                unparsed=item.unparsed,
            )
            for item in base.per_source
        ],
        reference_run_id=run_id,
        jobs_in_reference_run=run.jobs_total,
        jobs_with_skills=run.jobs_with_skills,
        skill_coverage_pct=round(skill_coverage_pct, 2),
        embedded_jobs=int(embedded_jobs or 0),
        embedding_coverage_pct=round(embedding_coverage_pct, 2),
        ontology_skill_count=len(load_ontology()),
        distinct_skills_matched=int(distinct_skills_matched or 0),
    )


@router.get("/stats/skills", response_model=StatsSkillsResponse)
async def stats_skills(
    run_id: int | None = Query(
        None, description="Reference enrichment run; omit for the latest available run"
    ),
    country: str | None = Query(None, description="ISO2 country code, e.g. fr, de, gb"),
    seniority: str | None = Query(
        None, description="junior|senior|lead|manager|internship|apprenticeship"
    ),
    limit: int = Query(30, gt=0, le=200),
    session: AsyncSession = Depends(get_db_session),
) -> StatsSkillsResponse:
    try:
        run_id = await resolve_run_id(session, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    optional_filters = []
    if country:
        optional_filters.append(Job.country == country.casefold())
    if seniority:
        pattern = _SENIORITY_PATTERNS.get(seniority.casefold())
        if pattern is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown seniority filter: {seniority!r}. "
                f"Choose one of {sorted(_SENIORITY_PATTERNS)}.",
            )
        optional_filters.append(or_(Job.title.op("~*")(pattern), Job.description.op("~*")(pattern)))

    total_considered = await session.scalar(
        select(func.count(func.distinct(Job.id)))
        .select_from(Job)
        .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
        .where(
            JobEnrichmentResult.run_id == run_id,
            JobEnrichmentResult.status == "success",
            *optional_filters,
        )
    )

    skill_query = (
        select(Skill.canonical, Skill.category, func.count(func.distinct(JobSkill.job_id)))
        .select_from(JobSkill)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id)
    )
    if optional_filters:
        skill_query = skill_query.join(Job, Job.id == JobSkill.job_id).where(*optional_filters)
    skill_query = (
        skill_query.group_by(Skill.canonical, Skill.category)
        .order_by(func.count(func.distinct(JobSkill.job_id)).desc(), Skill.canonical)
        .limit(limit)
    )
    rows = (await session.execute(skill_query)).all()

    considered = int(total_considered or 0)
    skills = [
        SkillDemandItem(
            canonical_skill=canonical,
            category=category,
            job_count=int(job_count),
            pct_of_considered_jobs=(
                round(int(job_count) / considered * 100, 2) if considered else 0.0
            ),
        )
        for canonical, category, job_count in rows
    ]

    return StatsSkillsResponse(
        run_id=run_id,
        country=country,
        seniority=seniority,
        total_jobs_considered=considered,
        skills=skills,
    )


__all__ = ["router"]
