"""Deterministic CV-to-job ranking against one persisted enrichment run."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.cv.attributes import extract_job_attributes
from jobmarket.cv.profile import (
    DEFAULT_RECOMMENDATION_RUN_ID,
    SCORING_VERSION,
    VALIDATION_RUN_WARNING,
    CvProfile,
    JobAttributes,
    JobRecommendation,
    RecommendationPenalty,
    ScoreComponents,
)
from jobmarket.db.models import Company, EnrichmentRun, Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.embeddings.cv import embed_cv_profile
from jobmarket.embeddings.encoder import EmbeddingEncoder
from jobmarket.embeddings.repository import SemanticSearchFilters, search_jobs_by_embedding
from jobmarket.skills.ontology import load_ontology, normalize_alias

SCORE_WEIGHTS = {
    "skills": 0.50,
    "experience": 0.15,
    "career_level": 0.15,
    "domain": 0.10,
    "job_type": 0.05,
    "location": 0.05,
}
SKILL_SCORE_WEIGHTS = {
    "coverage": 0.55,
    "relevance": 0.20,
    "richness": 0.10,
    "specificity": 0.15,
}
JOB_MATCHING_FORMULA = (
    "final_score = 100 * max(0, "
    "0.50*skill_score + 0.15*experience_score + 0.15*career_level_score + "
    "0.10*domain_score + 0.05*job_type_score + 0.05*location_score - penalty_total)"
)
SKILL_MATCHING_FORMULA = (
    "skill_score = 0.55*coverage + 0.20*cv_relevance + "
    "0.10*job_skill_richness + 0.15*skill_specificity"
)
GENERIC_SKILLS = {
    "Artificial Intelligence",
    "Data Science",
    "Machine Learning",
    "Software Development",
    "Cloud Computing",
    "Backend Development",
}
TECHNICAL_RECOMMENDATION_DOMAINS = {
    "AI",
    "Data Science",
    "Machine Learning",
    "NLP",
    "RAG",
    "Generative AI",
    "Computer Vision",
    "Software Engineering",
    "Backend",
    "Cloud",
    "DevOps",
    "Security",
    "Data Engineering",
}
BUSINESS_PROFILE_DOMAINS = {
    "Management",
    "Customer Service",
    "Customer Relationship Management",
    "Sales",
    "Insurance",
    "Communication",
    "Digital Transformation",
    "Commercial Support",
    "Administrative Operations",
    "Stock Management",
}

SPECIALIZED_SKILLS = {
    "Retrieval-Augmented Generation",
    "Large Language Models",
    "Natural Language Processing",
    "Deep Learning",
    "Generative AI",
    "Computer Vision",
    "PostgreSQL",
    "MongoDB",
    "FastAPI",
    "scikit-learn",
    "Power BI",
    "AWS",
    "Azure",
    "GCP",
    "Kubernetes",
}
_CAREER_RANK = {
    "student": 0,
    "internship": 0,
    "apprenticeship": 0,
    "junior": 1,
    "mid": 2,
    "consultant": 2,
    "senior": 3,
    "lead": 4,
    "principal": 4,
    "manager": 4,
    "unknown": -1,
}
_RELATED_DOMAINS = {
    "AI": {"Data Science", "Machine Learning", "Generative AI", "NLP", "RAG", "Computer Vision"},
    "Machine Learning": {"Data Science", "AI", "Deep Learning", "Computer Vision", "NLP"},
    "Data Science": {"Machine Learning", "AI", "Data Engineering"},
    "Generative AI": {"AI", "NLP", "RAG", "Machine Learning"},
    "NLP": {"AI", "Generative AI", "RAG", "Machine Learning"},
    "RAG": {"Generative AI", "NLP", "AI"},
    "Data Engineering": {"Data Science", "Cloud"},
    "Cloud": {"DevOps", "Data Engineering"},
}
SkillSpecificity = Literal["generic", "standard", "specialized"]
RecommendationMode = Literal["lexical", "semantic", "hybrid"]
DEFAULT_HYBRID_ALPHA = 0.60
DEFAULT_LEXICAL_POOL_SIZE = 200
DEFAULT_SEMANTIC_POOL_SIZE = 200


async def resolve_run_id(session: AsyncSession, run_id: int | None) -> int:
    """Resolve `None` to the most recent usable matcher run; an explicit `run_id`
    passes through unchanged (still validated downstream, e.g. by
    `recommend_jobs_for_cv`).

    "Usable" matches the acceptance set already enforced there: matcher runs with
    status `success` or `partial`. Every API route that takes a `run_id` should call
    this instead of defaulting to a hardcoded literal — a hardcoded default silently
    keeps serving whatever snapshot existed when that literal was written, forever,
    even after a fresh ingest + enrichment run produces newer data (see
    docs/LIMITATIONS.md — this was found live: a fresh ingest produced run 1592 with
    postings as recent as the same day, but the API kept serving run 342 from three
    weeks earlier because `API_DEFAULT_RUN_ID = 342` was a literal, not a query).
    """
    if run_id is not None:
        return run_id
    latest = await session.execute(
        select(func.max(EnrichmentRun.id)).where(
            EnrichmentRun.run_type == "matcher",
            EnrichmentRun.status.in_(["success", "partial"]),
        )
    )
    resolved = latest.scalar_one_or_none()
    if resolved is None:
        raise ValueError("no usable matcher enrichment run exists")
    return resolved


async def recommend_jobs_for_cv(
    session: AsyncSession,
    profile: CvProfile,
    *,
    run_id: int = DEFAULT_RECOMMENDATION_RUN_ID,
    top: int = 20,
    mode: RecommendationMode = "lexical",
    alpha: float = DEFAULT_HYBRID_ALPHA,
    encoder: EmbeddingEncoder | None = None,
    lexical_pool_size: int = DEFAULT_LEXICAL_POOL_SIZE,
    semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE,
) -> list[JobRecommendation]:
    """Rank jobs for a CV profile using lexical, semantic, or hybrid retrieval."""
    _validate_recommendation_options(mode=mode, alpha=alpha)
    run = await session.get(EnrichmentRun, run_id)
    if run is None:
        raise ValueError(f"enrichment run {run_id} does not exist")
    if run.run_type != "matcher" or run.status not in {"success", "partial"}:
        raise ValueError(f"enrichment run {run_id} is not an approved matcher run")

    if mode == "lexical":
        return await _recommend_lexical(session, profile, run_id=run_id, top=top)

    cv_skills = {skill.canonical_skill for skill in profile.skills}
    cv_skill_count = len(cv_skills)
    domain_covered_skills = frozenset(_domain_covered_skills(profile))
    lexical_ids = await _lexical_candidate_ids(
        session, run_id=run_id, cv_skills=cv_skills, limit=lexical_pool_size
    )
    try:
        cv_embedding = embed_cv_profile(profile, encoder=encoder)
        semantic_hits = await search_jobs_by_embedding(
            session,
            cv_embedding,
            limit=semantic_pool_size,
            filters=SemanticSearchFilters(
                country=profile.candidate.preferred_country,
                work_mode="remote" if profile.candidate.remote_preference is True else None,
            ),
        )
    except Exception:
        if mode == "hybrid":
            return await _recommend_lexical(session, profile, run_id=run_id, top=top)
        raise
    semantic_by_id = {hit.job_id: hit for hit in semantic_hits}
    candidate_ids = sorted(set(lexical_ids) | set(semantic_by_id))
    bundles = await _job_skill_bundles_for_ids(session, run_id=run_id, job_ids=candidate_ids)
    recommendations: list[JobRecommendation] = []
    for bundle in bundles.values():
        if _suppress_bundle_for_profile(profile, bundle, cv_skills):
            continue
        in_lexical = bundle.job_id in lexical_ids
        in_semantic = bundle.job_id in semantic_by_id
        source = "both" if in_lexical and in_semantic else "semantic" if in_semantic else "lexical"
        hit = semantic_by_id.get(bundle.job_id)
        # None (not 0.0) for a job outside the semantic retrieval pool: "no semantic
        # score was computed" is not the same fact as "semantic similarity is zero",
        # and must not be blended as if it were — see _score_job.
        semantic_score = hit.semantic_score if hit is not None else None
        recommendation = _score_job(
            bundle,
            profile=profile,
            cv_skills=cv_skills,
            cv_skill_count=cv_skill_count,
            run_id=run_id,
            mode=mode,
            semantic_score=semantic_score,
            alpha=(
                alpha * profile.extraction_quality.extraction_confidence_score
                if mode == "hybrid"
                else 0.0
            ),
            retrieval_source=source,
            semantic_rank=hit.semantic_rank if hit is not None else None,
            domain_covered_skills=domain_covered_skills,
        )
        if not _should_suppress_recommendation(recommendation):
            recommendations.append(recommendation)
    recommendations.sort(key=_ranking_key)
    return recommendations[:top]


async def _recommend_lexical(
    session: AsyncSession,
    profile: CvProfile,
    *,
    run_id: int,
    top: int,
) -> list[JobRecommendation]:
    cv_skills = {skill.canonical_skill for skill in profile.skills}
    cv_skill_count = len(cv_skills)
    domain_covered_skills = frozenset(_domain_covered_skills(profile))
    job_rows = await _job_skill_rows(session, run_id)
    by_job = _bundles_from_rows(job_rows)
    recommendations: list[JobRecommendation] = []
    for bundle in by_job.values():
        if not bundle.skills or _suppress_bundle_for_profile(profile, bundle, cv_skills):
            continue
        recommendation = _score_job(
            bundle,
            profile=profile,
            cv_skills=cv_skills,
            cv_skill_count=cv_skill_count,
            run_id=run_id,
            domain_covered_skills=domain_covered_skills,
        )
        if not _should_suppress_recommendation(recommendation):
            recommendations.append(recommendation)
    recommendations.sort(key=_ranking_key)
    return recommendations[:top]


def _should_suppress_recommendation(recommendation: JobRecommendation) -> bool:
    return any(
        penalty.code == "business_profile_technical_role_mismatch"
        for penalty in recommendation.penalties
    )


def _suppress_bundle_for_profile(
    profile: CvProfile, bundle: _JobSkillBundle, cv_skills: set[str]
) -> bool:
    if not _business_profile_without_technical_evidence(profile, cv_skills):
        return False
    return _technical_bundle_signal(bundle)


def _business_profile_without_technical_evidence(profile: CvProfile, cv_skills: set[str]) -> bool:
    candidate_domains = set(profile.candidate.preferred_domains)
    has_business_profile = bool(candidate_domains & BUSINESS_PROFILE_DOMAINS)
    technical_evidence = cv_skills & (
        GENERIC_SKILLS | SPECIALIZED_SKILLS | TECHNICAL_RECOMMENDATION_DOMAINS
    )
    return has_business_profile and not technical_evidence


def _technical_bundle_signal(bundle: _JobSkillBundle) -> bool:
    title = normalize_alias(bundle.title)
    technical_title = re.search(
        r"\b(data|developer|developpeur|full stack|software|logiciel|"
        r"technical|technology|engineer|ingenieur|architect|architecture|"
        r"technico|integrateur|int?grateur|database|tester|testeur|"
        r"\bsw\b|it)\b",
        title,
        re.IGNORECASE,
    )
    technical_skills = bool(bundle.skills & (GENERIC_SKILLS | SPECIALIZED_SKILLS))
    return bool(technical_title or technical_skills)


def _validate_recommendation_options(*, mode: RecommendationMode, alpha: float) -> None:
    if mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError("mode must be lexical, semantic, or hybrid")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")


def _bundles_from_rows(job_rows: list[dict[str, Any]]) -> dict[int, _JobSkillBundle]:
    by_job: dict[int, _JobSkillBundle] = {}
    for row in job_rows:
        job_id = int(row["job_id"])
        bundle = by_job.setdefault(
            job_id,
            _JobSkillBundle(
                job_id=job_id,
                title=str(row["title"]),
                description=str(row["description"]),
                company=str(row["company"]),
                location=row["location"],
                country=row["country"],
                city=row["city"],
                contract_type=row["contract_type"],
                is_remote=row["is_remote"],
                source_url=row.get("source_url"),
                posted_at=row.get("posted_at"),
                skills=set(),
            ),
        )
        canonical = row.get("canonical")
        if canonical is not None:
            bundle.skills.add(str(canonical))
    return by_job


async def _lexical_candidate_ids(
    session: AsyncSession,
    *,
    run_id: int,
    cv_skills: set[str],
    limit: int,
) -> list[int]:
    if limit <= 0:
        return []
    if not cv_skills:
        rows = await session.execute(
            select(Job.id)
            .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
            .where(
                JobEnrichmentResult.run_id == run_id,
                JobEnrichmentResult.status == "success",
                JobEnrichmentResult.skill_count > 0,
            )
            .order_by(Job.id)
            .limit(limit)
        )
        return [int(row[0]) for row in rows.all()]
    rows = await session.execute(
        select(Job.id)
        .join(JobSkill, (JobSkill.job_id == Job.id) & (JobSkill.run_id == run_id))
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(Skill.canonical.in_(sorted(cv_skills)))
        .group_by(Job.id)
        .order_by(func.count().desc(), Job.id.asc())
        .limit(limit)
    )
    return [int(row[0]) for row in rows.all()]


async def _job_skill_bundles_for_ids(
    session: AsyncSession,
    *,
    run_id: int,
    job_ids: list[int],
) -> dict[int, _JobSkillBundle]:
    if not job_ids:
        return {}
    rows = await session.execute(
        select(
            Job.id.label("job_id"),
            Job.title.label("title"),
            Job.description.label("description"),
            Company.name.label("company"),
            Job.location_raw.label("location"),
            Job.country.label("country"),
            Job.city.label("city"),
            Job.contract_type.label("contract_type"),
            Job.is_remote.label("is_remote"),
            Job.url.label("source_url"),
            Job.posted_at.label("posted_at"),
            Skill.canonical.label("canonical"),
        )
        .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
        .join(Company, Company.id == Job.company_id)
        .outerjoin(JobSkill, (JobSkill.job_id == Job.id) & (JobSkill.run_id == run_id))
        .outerjoin(Skill, Skill.id == JobSkill.skill_id)
        .where(
            Job.id.in_(job_ids),
            JobEnrichmentResult.run_id == run_id,
            JobEnrichmentResult.status == "success",
        )
        .order_by(Job.id, Skill.canonical)
    )
    return _bundles_from_rows([dict(row) for row in rows.mappings().all()])


def validation_run_warning(run_id: int) -> str | None:
    """Return the MVP warning for the default validation run."""
    if run_id == DEFAULT_RECOMMENDATION_RUN_ID:
        return VALIDATION_RUN_WARNING
    return None


def score_values(
    *,
    matched_skill_count: int,
    job_skill_count: int,
    cv_skill_count: int,
) -> tuple[float, float, float]:
    """Calculate coverage, CV relevance, and v3 blended skill score."""
    coverage_score = matched_skill_count / job_skill_count if job_skill_count else 0.0
    cv_overlap_score = matched_skill_count / cv_skill_count if cv_skill_count else 0.0
    richness = skill_profile_confidence(job_skill_count)
    skill_score = (
        SKILL_SCORE_WEIGHTS["coverage"] * coverage_score
        + SKILL_SCORE_WEIGHTS["relevance"] * cv_overlap_score
        + SKILL_SCORE_WEIGHTS["richness"] * richness
        + SKILL_SCORE_WEIGHTS["specificity"] * 0.65
    )
    return coverage_score, cv_overlap_score, round(skill_score, 4)


def skill_specificity(canonical: str) -> SkillSpecificity:
    """Classify a canonical skill's recommendation specificity without changing ontology."""
    if canonical in GENERIC_SKILLS:
        return "generic"
    if canonical in SPECIALIZED_SKILLS:
        return "specialized"
    return "standard"


def skill_profile_confidence(job_skill_count: int) -> float:
    """Small deterministic confidence signal based on detected job-skill richness."""
    if job_skill_count <= 0:
        return 0.0
    if job_skill_count == 1:
        return 0.60
    if job_skill_count == 2:
        return 0.75
    if job_skill_count <= 5:
        return 0.90
    return 1.0


class _JobSkillBundle:
    def __init__(
        self,
        *,
        job_id: int,
        title: str,
        description: str,
        company: str,
        location: str | None,
        country: str | None,
        city: str | None,
        contract_type: str | None,
        is_remote: bool | None,
        source_url: str | None,
        posted_at: datetime | None = None,
        skills: set[str],
    ) -> None:
        self.job_id = job_id
        self.title = title
        self.description = description
        self.company = company
        self.location = location
        self.country = country
        self.city = city
        self.contract_type = contract_type
        self.is_remote = is_remote
        self.source_url = source_url
        self.posted_at = posted_at
        self.skills = skills


async def _job_skill_rows(session: AsyncSession, run_id: int) -> list[dict[str, Any]]:
    rows = await session.execute(
        select(
            Job.id.label("job_id"),
            Job.title.label("title"),
            Job.description.label("description"),
            Company.name.label("company"),
            Job.location_raw.label("location"),
            Job.country.label("country"),
            Job.city.label("city"),
            Job.contract_type.label("contract_type"),
            Job.is_remote.label("is_remote"),
            Job.url.label("source_url"),
            Job.posted_at.label("posted_at"),
            Skill.canonical.label("canonical"),
        )
        .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
        .join(Company, Company.id == Job.company_id)
        .join(JobSkill, (JobSkill.job_id == Job.id) & (JobSkill.run_id == run_id))
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(
            JobEnrichmentResult.run_id == run_id,
            JobEnrichmentResult.status == "success",
            JobEnrichmentResult.skill_count > 0,
        )
        .order_by(Job.id, Skill.canonical)
    )
    return [dict(row) for row in rows.mappings().all()]


def _domain_covered_skills(profile: CvProfile) -> set[str]:
    """Canonical skill names already demonstrated via the candidate's domain/role
    signals (e.g. "AI" resolves to the ontology's domain-category skill "Artificial
    Intelligence", since "ai" is a registered alias of that canonical entry).

    Used ONLY to keep a domain/role the candidate already has from being reported as
    a missing-skill gap — deliberately NOT folded into `cv_skills` itself, which also
    drives `matched`/the coverage and skill scores. Treating a domain label as
    equivalent to an explicit, evidenced skill for scoring purposes would be a much
    bigger claim than "don't call this a gap when the candidate's own profile already
    names it" — out of scope here; this stays scoped to gap-reporting.
    """
    ontology = load_ontology()
    raw_labels = (
        set(profile.candidate.preferred_domains)
        | {role.value for role in profile.inferred_roles}
        | {role.value for role in profile.inferred_domains}
    )
    covered = set()
    for label in raw_labels:
        canonical = ontology.resolve(label)
        if canonical is not None:
            covered.add(canonical)
    return covered


def _score_job(
    bundle: _JobSkillBundle,
    *,
    profile: CvProfile,
    cv_skills: set[str],
    cv_skill_count: int,
    run_id: int,
    mode: RecommendationMode = "lexical",
    semantic_score: float | None = None,
    alpha: float = DEFAULT_HYBRID_ALPHA,
    retrieval_source: str = "lexical",
    semantic_rank: int | None = None,
    domain_covered_skills: frozenset[str] = frozenset(),
) -> JobRecommendation:
    job_skills = sorted(bundle.skills)
    matched = sorted(cv_skills & bundle.skills)
    missing = sorted(bundle.skills - cv_skills - domain_covered_skills)
    skill_parts = _skill_score_components(
        matched=matched,
        job_skills=job_skills,
        cv_skill_count=cv_skill_count,
    )
    job_attributes = extract_job_attributes(
        title=bundle.title,
        description=bundle.description,
        contract_type=bundle.contract_type,
        is_remote=bundle.is_remote,
        country=bundle.country,
        city=bundle.city,
        skill_canonicals=bundle.skills,
    )
    experience_score, experience_gap = _experience_score(profile, job_attributes)
    career_score = _career_level_score(profile, job_attributes)
    job_type_score = _job_type_score(profile, job_attributes)
    domain_score = _domain_score(profile, job_attributes)
    location_score, location_status = _location_score(profile, job_attributes)
    penalties = _penalties(profile, job_attributes, experience_gap)
    penalty_total = min(0.75, sum(penalty.amount for penalty in penalties))
    weighted = (
        SCORE_WEIGHTS["skills"] * skill_parts["skill_score"]
        + SCORE_WEIGHTS["experience"] * experience_score
        + SCORE_WEIGHTS["career_level"] * career_score
        + SCORE_WEIGHTS["domain"] * domain_score
        + SCORE_WEIGHTS["job_type"] * job_type_score
        + SCORE_WEIGHTS["location"] * location_score
    )
    lexical_score = max(0.0, weighted)
    semantic_value = None if semantic_score is None else max(0.0, min(1.0, semantic_score))
    hybrid_before_penalty = None
    if mode == "lexical":
        final_score = round(max(0.0, weighted - penalty_total) * 100, 2)
    elif semantic_value is None:
        # This job was never in the semantic retrieval pool: there is no evidence to
        # blend, not evidence of zero similarity. Score it on lexical alone rather
        # than crediting/debiting a phantom 0.0 into the weighted blend.
        hybrid_before_penalty = lexical_score
        final_score = round(max(0.0, hybrid_before_penalty - penalty_total) * 100, 2)
    else:
        effective_alpha = alpha if mode == "hybrid" else 0.0
        hybrid_before_penalty = (
            effective_alpha * lexical_score + (1.0 - effective_alpha) * semantic_value
        )
        final_score = round(max(0.0, hybrid_before_penalty - penalty_total) * 100, 2)
    components = ScoreComponents(
        skill_score=round(skill_parts["skill_score"], 4),
        coverage_component=round(skill_parts["coverage"], 4),
        relevance_component=round(skill_parts["relevance"], 4),
        richness_component=round(skill_parts["richness"], 4),
        specificity_component=round(skill_parts["specificity"], 4),
        skill_profile_confidence=round(skill_parts["richness"], 4),
        specialized_matched_count=int(skill_parts["specialized_count"]),
        generic_matched_count=int(skill_parts["generic_count"]),
        experience_score=round(experience_score, 4),
        career_level_score=round(career_score, 4),
        job_type_score=round(job_type_score, 4),
        domain_score=round(domain_score, 4),
        location_score=round(location_score, 4),
        location_status=location_status,
        weighted_score_before_penalty=round(weighted, 4),
        penalty_total=round(penalty_total, 4),
        lexical_score=round(lexical_score, 4),
        semantic_score=round(semantic_value, 4) if semantic_value is not None else None,
        alpha=alpha if mode == "hybrid" else 0.0 if mode == "semantic" else None,
        hybrid_score_before_penalty=(
            round(hybrid_before_penalty, 4) if hybrid_before_penalty is not None else None
        ),
    )
    specialized_matched = [skill for skill in matched if skill_specificity(skill) == "specialized"]
    generic_matched = [skill for skill in matched if skill_specificity(skill) == "generic"]
    tie_break_factors = {
        "final_score": final_score,
        "semantic_score": round(semantic_value, 4) if semantic_value is not None else 0.0,
        "skill_coverage": round(skill_parts["coverage"], 4),
        "specialized_skill_overlap_count": len(specialized_matched),
        "total_matched_skill_count": len(matched),
        "job_skill_richness": len(job_skills),
        "career_compatibility": round(career_score, 4),
        "experience_compatibility": round(experience_score, 4),
        "job_id": bundle.job_id,
    }
    return JobRecommendation(
        job_id=bundle.job_id,
        title=bundle.title,
        company=bundle.company,
        location=bundle.location,
        contract_type=bundle.contract_type,
        source_url=bundle.source_url,
        posted_at=bundle.posted_at,
        final_score=final_score,
        coverage_score=skill_parts["coverage"],
        cv_overlap_score=skill_parts["relevance"],
        cv_skill_count=cv_skill_count,
        job_skill_count=len(job_skills),
        matched_skill_count=len(matched),
        matched_skills=matched,
        missing_skills=missing,
        specialized_matched_skills=specialized_matched,
        generic_matched_skills=generic_matched,
        enrichment_run_id=run_id,
        scoring_version=SCORING_VERSION,
        explanation=_explanation(
            matched_count=len(matched),
            job_skill_count=len(job_skills),
            components=components,
            job_attributes=job_attributes,
            penalties=penalties,
            experience_gap=experience_gap,
        ),
        job_attributes=job_attributes,
        score_components=components,
        penalties=penalties,
        experience_gap_years=experience_gap,
        tie_break_factors=tie_break_factors,
        semantic_score=round(semantic_value, 4) if semantic_value is not None else None,
        semantic_rank=semantic_rank,
        retrieval_source=retrieval_source,
        hybrid_score_before_penalties=(
            round(hybrid_before_penalty, 4) if hybrid_before_penalty is not None else None
        ),
    )


def _ranking_key(item: JobRecommendation) -> tuple[float, float, int, int, int, float, float, int]:
    factors = item.tie_break_factors
    return (
        -item.final_score,
        -float(factors.get("skill_coverage", item.coverage_score)),
        -int(factors.get("specialized_skill_overlap_count", 0)),
        -item.matched_skill_count,
        -item.job_skill_count,
        -float(factors.get("career_compatibility", 0)),
        -float(factors.get("experience_compatibility", 0)),
        item.job_id,
    )


def _skill_score_components(
    *,
    matched: list[str],
    job_skills: list[str],
    cv_skill_count: int,
) -> dict[str, float]:
    job_skill_count = len(job_skills)
    coverage = len(matched) / job_skill_count if job_skill_count else 0.0
    relevance = len(matched) / cv_skill_count if cv_skill_count else 0.0
    richness = skill_profile_confidence(job_skill_count)
    specificity = _matched_specificity_score(matched)
    score = (
        SKILL_SCORE_WEIGHTS["coverage"] * coverage
        + SKILL_SCORE_WEIGHTS["relevance"] * relevance
        + SKILL_SCORE_WEIGHTS["richness"] * richness
        + SKILL_SCORE_WEIGHTS["specificity"] * specificity
    )
    specialized_count = sum(1 for skill in matched if skill_specificity(skill) == "specialized")
    generic_count = sum(1 for skill in matched if skill_specificity(skill) == "generic")
    return {
        "coverage": coverage,
        "relevance": relevance,
        "richness": richness,
        "specificity": specificity,
        "skill_score": min(1.0, score),
        "specialized_count": float(specialized_count),
        "generic_count": float(generic_count),
    }


def _matched_specificity_score(matched: list[str]) -> float:
    if not matched:
        return 0.0
    values = [_specificity_value(skill_specificity(skill)) for skill in matched]
    return sum(values) / len(values)


def _specificity_value(specificity: SkillSpecificity) -> float:
    if specificity == "specialized":
        return 1.0
    if specificity == "generic":
        return 0.35
    return 0.65


def _experience_score(profile: CvProfile, job: JobAttributes) -> tuple[float, float | None]:
    required = job.experience_min_years
    candidate_years = profile.candidate.experience.total_years
    if required is None:
        return 0.9, None
    if candidate_years is None:
        return 0.55, required
    if candidate_years >= required:
        return 1.0, 0.0
    gap = round(required - candidate_years, 2)
    return max(0.15, 1.0 - (gap / max(required, 1.0))), gap


def _career_level_score(profile: CvProfile, job: JobAttributes) -> float:
    candidate = profile.candidate.career_level
    target = job.career_level
    if target == "unknown":
        return 0.85
    if candidate == "unknown":
        return 0.65
    if target in {"internship", "apprenticeship"}:
        return 1.0 if candidate in {"student", "internship", "apprenticeship", "junior"} else 0.55
    candidate_rank = _CAREER_RANK[candidate]
    target_rank = _CAREER_RANK[target]
    if candidate_rank >= target_rank:
        return 1.0
    gap = target_rank - candidate_rank
    if gap == 1:
        return 0.7
    if gap == 2:
        return 0.35
    return 0.1


def _job_type_score(profile: CvProfile, job: JobAttributes) -> float:
    candidate = profile.candidate
    job_types = set(job.job_types)
    if not job_types:
        return 0.65
    if job_types & {"stage", "apprenticeship"}:
        compatible_entry_levels = {"student", "internship", "apprenticeship"}
        return 1.0 if candidate.career_level in compatible_entry_levels else 0.55
    if "freelance" in job_types:
        if candidate.experience.total_years is None or candidate.experience.total_years < 2:
            return 0.35
        return 0.85
    return 0.75


def _domain_score(profile: CvProfile, job: JobAttributes) -> float:
    job_domains = set(job.domains)
    candidate_domains = set(profile.candidate.preferred_domains)
    if not job_domains:
        return 0.45
    if not candidate_domains:
        return 0.35
    scores = [_best_domain_match_score(job_domain, candidate_domains) for job_domain in job_domains]
    return round(sum(scores) / len(scores), 4)


def _best_domain_match_score(job_domain: str, candidate_domains: set[str]) -> float:
    if job_domain in candidate_domains:
        return 1.0
    related = _RELATED_DOMAINS.get(job_domain, set())
    if related & candidate_domains:
        return 0.75
    if job_domain == "AI" or "AI" in candidate_domains:
        return 0.55
    return 0.0


def _location_score(profile: CvProfile, job: JobAttributes) -> tuple[float, str]:
    candidate = profile.candidate
    candidate_city = _norm_location(candidate.preferred_city)
    candidate_country = _norm_location(candidate.preferred_country)
    job_city = _norm_location(job.city)
    job_country = _norm_location(job.country)
    if candidate.remote_preference is True and job.remote_mode in {"remote", "hybrid"}:
        return 0.90, "remote_compatible"
    if candidate_city and job_city and candidate_city == job_city:
        return 1.00, "exact_match"
    if candidate_country and job_country and candidate_country == job_country:
        return 0.75, "compatible"
    if not candidate_city and not candidate_country and candidate.remote_preference is None:
        return 0.50, "unknown_candidate_preference"
    if not job_city and not job_country and job.remote_mode is None:
        return 0.50, "unknown_job_location"
    return 0.25, "mismatch"


def _norm_location(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_alias(value).strip()
    return normalized or None


def _penalties(
    profile: CvProfile,
    job: JobAttributes,
    experience_gap: float | None,
) -> list[RecommendationPenalty]:
    penalties: list[RecommendationPenalty] = []
    candidate = profile.candidate
    senior_levels = {"senior", "lead", "principal", "manager"}
    if candidate.career_level == "student" and job.career_level in senior_levels:
        penalties.append(
            RecommendationPenalty(
                code="student_senior_mismatch",
                amount=0.3,
                explanation=f"Student profile matched to {job.career_level} role.",
            )
        )
    if (
        candidate.career_level in {"student", "internship"}
        and job.experience_min_years is not None
        and job.experience_min_years >= 5
    ):
        penalties.append(
            RecommendationPenalty(
                code="large_experience_requirement",
                amount=0.25,
                explanation=f"Requires {job.experience_min_years:g}+ years experience.",
            )
        )
    if "freelance" in job.job_types and (
        candidate.experience.total_years is None or candidate.experience.total_years < 2
    ):
        penalties.append(
            RecommendationPenalty(
                code="freelance_low_experience",
                amount=0.25,
                explanation="Freelance role with limited explicit professional experience.",
            )
        )

    if _business_profile_technical_role_mismatch(profile, job):
        penalties.append(
            RecommendationPenalty(
                code="business_profile_technical_role_mismatch",
                amount=0.45,
                explanation=(
                    "Business/management CV has no explicit technical skill evidence "
                    "for this technical/data role."
                ),
            )
        )
    if job.experience_min_years is not None and candidate.experience.total_years is None:
        penalties.append(
            RecommendationPenalty(
                code="unknown_candidate_experience",
                amount=0.08,
                explanation="CV has no explicit total professional experience duration.",
            )
        )
    elif experience_gap is not None and experience_gap > 0:
        penalties.append(
            RecommendationPenalty(
                code="experience_gap",
                amount=min(0.25, 0.06 * experience_gap),
                explanation=f"Experience gap of {experience_gap:g} years.",
            )
        )
    return penalties


def _business_profile_technical_role_mismatch(profile: CvProfile, job: JobAttributes) -> bool:
    candidate_domains = set(profile.candidate.preferred_domains)
    job_domains = set(job.domains)
    cv_skills = {skill.canonical_skill for skill in profile.skills}
    has_business_profile = bool(candidate_domains & BUSINESS_PROFILE_DOMAINS)
    has_technical_evidence = bool(
        cv_skills & (GENERIC_SKILLS | SPECIALIZED_SKILLS | TECHNICAL_RECOMMENDATION_DOMAINS)
    )
    has_technical_job = bool(job_domains & TECHNICAL_RECOMMENDATION_DOMAINS)
    return has_business_profile and has_technical_job and not has_technical_evidence


def _explanation(
    *,
    matched_count: int,
    job_skill_count: int,
    components: ScoreComponents,
    job_attributes: JobAttributes,
    penalties: list[RecommendationPenalty],
    experience_gap: float | None,
) -> str:
    career_source = _format_first_provenance(job_attributes.career_level_provenance)
    type_source = _format_provenance(job_attributes.job_type_provenance)
    parts = [
        f"Matched {matched_count} of {job_skill_count} job skills",
        f"skill={components.skill_score:.2f}",
        f"coverage={components.coverage_component:.2f}",
        f"specificity={components.specificity_component:.2f}",
        f"skill_profile_confidence={components.skill_profile_confidence:.2f}",
        (
            f"career={components.career_level_score:.2f} "
            f"({job_attributes.career_level}; {career_source})"
        ),
        f"experience={components.experience_score:.2f}",
        f"job_type={components.job_type_score:.2f} ({type_source})",
        f"domain={components.domain_score:.2f}",
        f"location={components.location_score:.2f} ({components.location_status})",
    ]
    if experience_gap is not None and experience_gap > 0:
        parts.append(f"experience gap={experience_gap:g} years")
    if penalties:
        parts.append("penalties=" + "; ".join(penalty.code for penalty in penalties))
    else:
        parts.append("penalties=none")
    return "; ".join(parts) + "."


def _format_first_provenance(items: list[Any]) -> str:
    if not items:
        return "no explicit evidence"
    item = items[0]
    return f"{item.source}: {item.evidence!r}"


def _format_provenance(items: list[Any]) -> str:
    if not items:
        return "unknown"
    return ", ".join(f"{item.value} from {item.source}: {item.evidence!r}" for item in items)


__all__ = [
    "DEFAULT_RECOMMENDATION_RUN_ID",
    "GENERIC_SKILLS",
    "JOB_MATCHING_FORMULA",
    "SCORE_WEIGHTS",
    "SCORING_VERSION",
    "SKILL_MATCHING_FORMULA",
    "SKILL_SCORE_WEIGHTS",
    "SPECIALIZED_SKILLS",
    "recommend_jobs_for_cv",
    "resolve_run_id",
    "score_values",
    "skill_profile_confidence",
    "skill_specificity",
    "validation_run_warning",
]
