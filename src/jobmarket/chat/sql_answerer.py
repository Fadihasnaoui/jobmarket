"""Analytical/numeric chat answers, computed entirely from parameterized SQL.

The one rule this whole module exists to enforce: **no LLM ever computes or phrases
a number here**. A question is classified as analytical (see `router.py`), entities
(skill/country/seniority/contract_type/role title) are extracted deterministically
(`entities.py`, no LLM), a real parameterized query runs against the DB, and the
answer sentence is built with plain string formatting around the query's own result —
never generated text that could subtly misstate the number. If a question can't be
mapped to a query this module actually supports, it says so plainly instead of
guessing.

Reuses `api/stats_routes.py`'s exact query shapes (`_SENIORITY_PATTERNS`, the
run/job-enrichment-result join pattern) and `cv/matching.py::resolve_run_id` (always
the latest usable run, never a stale hardcoded one).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.api.stats_routes import _SENIORITY_PATTERNS
from jobmarket.chat.entities import ExtractedEntities, extract_entities
from jobmarket.cv.matching import resolve_run_id
from jobmarket.db.models import Job, JobEnrichmentResult, JobSkill, Skill
from jobmarket.skills.ontology import SkillsOntology

DEFAULT_TOP_SKILLS_LIMIT = 10

# Intent-detection patterns, English and French side by side -- users ask in either
# (confirmed live: French analytical phrasing is a real, common case, not an edge
# case). Checked most-specific-first so e.g. a salary question never falls through to
# a generic count. Kept separate from `router.py`'s `_ANALYTICAL_RE` on purpose: that
# regex only decides "is this analytical at all", this one decides *which* SQL query
# answers it -- the two can't share one pattern set without conflating them.
_AVG_SALARY_RE = re.compile(
    r"\baverage\b|\bmean\b|\btypical salary\b|salary (of|for|in)|"
    r"how much (does|do|is|are|can)|"
    r"\bmoyenne\b|\bsalaire\b|combien gagne",
    re.IGNORECASE,
)
_TOP_SKILLS_RE = re.compile(
    r"\btop \d*\s*skills\b|\bmost common\b|\bmost in-demand\b|\bmost requested\b|"
    r"\bmost popular\b|\bmost needed\b|"
    r"plus demand[ée]e?s?|plus recherch[ée]e?s?|plus courantes?|plus populaires?|"
    r"quelles? comp[ée]tences|top comp[ée]tences",
    re.IGNORECASE,
)
_PCT_RE = re.compile(
    r"%|percent|percentage|proportion|fraction|share of|"
    r"pourcentage|proportion de|part de",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"\bhow many\b|\bnumber of\b|\bcount of\b|\btotal (number|count)\b|"
    r"\bcombien\b|\bnombre de\b",
    re.IGNORECASE,
)
# Words that mean "job(s)" or "skill(s)" in either language -- used to decide whether
# a count question mentioning a skill wants the skill's demand count or the plain
# total job count (see `detect_sql_intent`'s docstring below).
_JOB_OR_SKILL_WORDS = (
    "skill",
    "job",
    "compétence",
    "competence",
    "offre",
    "poste",
)

# Fixed conversion table makes disclosed-salary averages comparable across currencies.
_FX_TO_EUR = {"eur": 1.00, "gbp": 1.17, "usd": 0.92}
# Country fallback retains valid salary ranges whose source omitted the currency code.
_COUNTRY_CURRENCY = {"gb": "gbp", "fr": "eur", "de": "eur", "nl": "eur", "es": "eur", "it": "eur"}
# Filters the RemoteOK 0/0 placeholder and obvious day/hour-rate contamination.
_SALARY_FLOOR_EUR = 2_000.0


class SqlIntent(StrEnum):
    SKILL_DEMAND = "skill_demand"
    TOP_SKILLS = "top_skills"
    AVG_SALARY = "avg_salary"
    JOB_COUNT = "job_count"


@dataclass(frozen=True)
class SqlAnswer:
    intent: SqlIntent
    answer: str
    entities: ExtractedEntities
    run_id: int
    data: dict[str, Any] = field(default_factory=dict)


def detect_sql_intent(question: str, entities: ExtractedEntities) -> SqlIntent:
    """Picks the specific SQL query, not just "this is analytical" (that's already
    decided by `router.py` before this runs). Order matters: salary and top-skills
    wording are checked before percentage/count wording so e.g. "average salary" or
    "most in-demand skills" never fall through to a generic count.
    """
    lowered = question.casefold()
    if _AVG_SALARY_RE.search(lowered):
        return SqlIntent.AVG_SALARY
    if _TOP_SKILLS_RE.search(lowered):
        return SqlIntent.TOP_SKILLS
    if _PCT_RE.search(lowered) and entities.skill is not None:
        return SqlIntent.SKILL_DEMAND
    if _COUNT_RE.search(lowered):
        mentions_job_or_skill = any(word in lowered for word in _JOB_OR_SKILL_WORDS)
        if entities.skill is not None and not mentions_job_or_skill:
            return SqlIntent.SKILL_DEMAND
        return SqlIntent.JOB_COUNT
    if entities.skill is not None:
        return SqlIntent.SKILL_DEMAND
    return SqlIntent.JOB_COUNT


def _entity_filters(entities: ExtractedEntities) -> list[Any]:
    filters: list[Any] = []
    if entities.country:
        filters.append(Job.country == entities.country)
    if entities.seniority:
        pattern = _SENIORITY_PATTERNS[entities.seniority]
        filters.append(or_(Job.title.op("~*")(pattern), Job.description.op("~*")(pattern)))
    if entities.contract_type:
        filters.append(Job.contract_type == entities.contract_type)
    if entities.role_title:
        filters.append(Job.title.ilike(f"%{entities.role_title}%"))
    return filters


def _describe_filters(entities: ExtractedEntities) -> str:
    parts: list[str] = []
    if entities.role_title:
        parts.append(f'title matching "{entities.role_title}"')
    if entities.country:
        parts.append(f"country={entities.country.upper()}")
    if entities.seniority:
        parts.append(f"seniority={entities.seniority}")
    if entities.contract_type:
        parts.append(f"contract_type={entities.contract_type}")
    return ", ".join(parts) if parts else "no filters (whole corpus)"


async def _job_count(session: AsyncSession, run_id: int, filters: list[Any]) -> int:
    result = await session.scalar(
        select(func.count(func.distinct(Job.id)))
        .select_from(Job)
        .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
        .where(
            JobEnrichmentResult.run_id == run_id,
            JobEnrichmentResult.status == "success",
            *filters,
        )
    )
    return int(result or 0)


async def answer_sql_question(
    session: AsyncSession, question: str, ontology: SkillsOntology, *, run_id: int | None = None
) -> SqlAnswer:
    entities = extract_entities(question, ontology)
    intent = detect_sql_intent(question, entities)
    resolved_run_id = await resolve_run_id(session, run_id)
    filters = _entity_filters(entities)
    filter_desc = _describe_filters(entities)

    if intent == SqlIntent.SKILL_DEMAND:
        return await _answer_skill_demand(session, resolved_run_id, entities, filters, filter_desc)
    if intent == SqlIntent.TOP_SKILLS:
        return await _answer_top_skills(session, resolved_run_id, entities, filters, filter_desc)
    if intent == SqlIntent.AVG_SALARY:
        return await _answer_avg_salary(session, resolved_run_id, entities, filters, filter_desc)
    return await _answer_job_count(session, resolved_run_id, entities, filters, filter_desc)


async def _answer_job_count(
    session: AsyncSession,
    run_id: int,
    entities: ExtractedEntities,
    filters: list[Any],
    filter_desc: str,
) -> SqlAnswer:
    count = await _job_count(session, run_id, filters)
    answer = f"There are {count:,} jobs in the corpus matching {filter_desc} (run {run_id})."
    return SqlAnswer(
        intent=SqlIntent.JOB_COUNT,
        answer=answer,
        entities=entities,
        run_id=run_id,
        data={"job_count": count, "filters": filter_desc},
    )


async def _answer_skill_demand(
    session: AsyncSession,
    run_id: int,
    entities: ExtractedEntities,
    filters: list[Any],
    filter_desc: str,
) -> SqlAnswer:
    if entities.skill is None:
        answer = (
            "I couldn't identify a specific skill in your question, so I can't compute "
            "a demand percentage. Try naming a skill directly, e.g. \"what percent of "
            'jobs need Docker?".'
        )
        return SqlAnswer(
            intent=SqlIntent.SKILL_DEMAND, answer=answer, entities=entities, run_id=run_id
        )

    total = await _job_count(session, run_id, filters)
    skill_query = (
        select(func.count(func.distinct(JobSkill.job_id)))
        .select_from(JobSkill)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .join(Job, Job.id == JobSkill.job_id)
        .where(JobSkill.run_id == run_id, Skill.canonical == entities.skill, *filters)
    )
    skill_count = int(await session.scalar(skill_query) or 0)
    pct = round(skill_count / total * 100, 1) if total else 0.0

    if total == 0:
        answer = f"No jobs matched {filter_desc}, so I can't compute a demand percentage."
    else:
        answer = (
            f'{pct}% of jobs matching {filter_desc} mention "{entities.skill}" '
            f"({skill_count:,} of {total:,} jobs, run {run_id})."
        )
    return SqlAnswer(
        intent=SqlIntent.SKILL_DEMAND,
        answer=answer,
        entities=entities,
        run_id=run_id,
        data={
            "skill": entities.skill,
            "skill_job_count": skill_count,
            "total_job_count": total,
            "pct": pct,
            "filters": filter_desc,
        },
    )


async def _answer_top_skills(
    session: AsyncSession,
    run_id: int,
    entities: ExtractedEntities,
    filters: list[Any],
    filter_desc: str,
    *,
    limit: int = DEFAULT_TOP_SKILLS_LIMIT,
) -> SqlAnswer:
    total = await _job_count(session, run_id, filters)
    skill_query = (
        select(Skill.canonical, func.count(func.distinct(JobSkill.job_id)))
        .select_from(JobSkill)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id)
    )
    if filters:
        skill_query = skill_query.join(Job, Job.id == JobSkill.job_id).where(*filters)
    skill_query = (
        skill_query.group_by(Skill.canonical)
        .order_by(func.count(func.distinct(JobSkill.job_id)).desc(), Skill.canonical)
        .limit(limit)
    )
    rows = (await session.execute(skill_query)).all()
    top_skills = [{"skill": canonical, "job_count": int(count)} for canonical, count in rows]

    if not top_skills:
        answer = f"No skill data found for jobs matching {filter_desc}."
    else:
        listed = ", ".join(f"{item['skill']} ({item['job_count']:,} jobs)" for item in top_skills)
        answer = (
            f"Top {len(top_skills)} skills for jobs matching {filter_desc} "
            f"(out of {total:,} jobs, run {run_id}): {listed}."
        )
    return SqlAnswer(
        intent=SqlIntent.TOP_SKILLS,
        answer=answer,
        entities=entities,
        run_id=run_id,
        data={"top_skills": top_skills, "total_job_count": total, "filters": filter_desc},
    )


def _apply_percentile_ceiling(values: list[float], *, percentile: float = 0.99) -> list[float]:
    if len(values) < 20:
        return values  # too few points for a percentile cutoff to mean anything
    ordered = sorted(values)
    cutoff = ordered[min(len(ordered) - 1, int(len(ordered) * percentile))]
    return [v for v in values if v <= cutoff]


async def _answer_avg_salary(
    session: AsyncSession,
    run_id: int,
    entities: ExtractedEntities,
    filters: list[Any],
    filter_desc: str,
) -> SqlAnswer:
    rows = (
        await session.execute(
            select(
                Job.salary_min, Job.salary_max, Job.salary_currency, Job.country, Job.salary_period
            )
            .select_from(Job)
            .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
            .where(
                JobEnrichmentResult.run_id == run_id,
                JobEnrichmentResult.status == "success",
                Job.salary_period == "year",
                Job.salary_min.isnot(None),
                Job.salary_max.isnot(None),
                ~((Job.salary_min == 0) & (Job.salary_max == 0)),
                *filters,
            )
        )
    ).all()

    above_floor: list[float] = []
    for salary_min, salary_max, currency, country, _period in rows:
        explicit = (currency or "").casefold()
        fallback = _COUNTRY_CURRENCY.get((country or "").casefold(), "")
        rate = _FX_TO_EUR.get(explicit or fallback)
        if rate is None:
            continue  # unsupported/unresolvable currency, skip rather than mis-convert
        mean_eur = float((salary_min + salary_max) / 2) * rate
        if mean_eur < _SALARY_FLOOR_EUR:
            continue  # Filter likely day/hour-rate contamination.
        above_floor.append(mean_eur)

    # Trim the extreme high end before calculating the disclosed-salary average.
    # this specific filtered population rather than reusing a global constant --
    # otherwise a handful of erroneous high values could skew a small filtered average.
    converted = _apply_percentile_ceiling(above_floor)

    if not converted:
        answer = (
            f"I don't have enough disclosed-salary data for jobs matching {filter_desc} "
            "to compute a reliable average."
        )
        return SqlAnswer(
            intent=SqlIntent.AVG_SALARY,
            answer=answer,
            entities=entities,
            run_id=run_id,
            data={"sample_size": 0, "filters": filter_desc},
        )

    avg_eur = round(sum(converted) / len(converted))
    answer = (
        f"The average disclosed annual salary for jobs matching {filter_desc} is "
        f"approximately EUR {avg_eur:,} (based on {len(converted):,} postings that "
        f"disclosed a real salary, run {run_id}) — this reflects only the self-selected "
        "subset of postings that stated a salary, not the whole market."
    )
    return SqlAnswer(
        intent=SqlIntent.AVG_SALARY,
        answer=answer,
        entities=entities,
        run_id=run_id,
        data={"avg_salary_eur": avg_eur, "sample_size": len(converted), "filters": filter_desc},
    )


__all__ = [
    "DEFAULT_TOP_SKILLS_LIMIT",
    "SqlAnswer",
    "SqlIntent",
    "answer_sql_question",
    "detect_sql_intent",
]
