"""Deterministic quality-audit exports for matcher enrichment runs."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import Job, JobEnrichmentResult, JobSkill, JobSource, RawJob, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.ontology import normalize_alias

TOP_SKILL_REVIEW_SEED = 7501
ZERO_SKILL_REVIEW_SEED = 7502
DESCRIPTION_EXCERPT_CHARS = 320
CONTEXT_CHARS = 80
COMMON_AMBIGUOUS_ALIASES = frozenset(
    {
        "ai",
        "bi",
        "c",
        "ci",
        "go",
        "it",
        "ml",
        "nlp",
        "qa",
        "r",
        "chef",
        "jira",
        "monday",
        "notion",
        "tableau",
    }
)


@dataclass(frozen=True)
class AuditExportResult:
    output_dir: Path
    top_skill_review_rows: int
    zero_skill_review_rows: int
    high_skill_count_review_rows: int
    ambiguous_alias_review_rows: int
    skill_frequency_rows: int
    evidence_integrity_violations: int


@dataclass(frozen=True)
class EvidenceRecord:
    run_id: int
    job_id: int
    canonical: str
    matched_alias: str
    evidence_text: str
    evidence_field: str
    start_char: int
    end_char: int
    title: str
    description: str
    source: str
    country: str


@dataclass(frozen=True)
class ZeroSkillRecord:
    run_id: int
    job_id: int
    title: str
    description: str
    source: str
    country: str


@dataclass(frozen=True)
class SkillCountRecord:
    run_id: int
    job_id: int
    skill_count: int
    title: str
    description: str
    source: str
    country: str
    detected_skills: str
    evidence: str


TOP_SKILL_REVIEW_FIELDS = [
    "run_id",
    "canonical",
    "job_id",
    "matched_alias",
    "evidence_text",
    "evidence_field",
    "start_char",
    "end_char",
    "title",
    "description_excerpt",
    "source",
    "country",
    "expected_skills",
    "false_positive",
    "reviewer_notes",
]
ZERO_SKILL_REVIEW_FIELDS = [
    "run_id",
    "job_id",
    "title",
    "description_excerpt",
    "source",
    "country",
    "expected_skills",
    "missed_skill",
    "ontology_gap",
    "reviewer_notes",
]
HIGH_SKILL_COUNT_REVIEW_FIELDS = [
    "run_id",
    "job_id",
    "skill_count",
    "title",
    "description_excerpt",
    "source",
    "country",
    "detected_skills",
    "evidence",
    "expected_skills",
    "false_positive_skills",
    "reviewer_notes",
]
AMBIGUOUS_ALIAS_REVIEW_FIELDS = [
    "run_id",
    "job_id",
    "canonical",
    "matched_alias",
    "evidence_text",
    "evidence_field",
    "start_char",
    "end_char",
    "context_before",
    "context_after",
    "title",
    "source",
    "country",
    "expected_skills",
    "false_positive",
    "reviewer_notes",
]
SKILL_FREQUENCY_FIELDS = ["metric", "key", "secondary_key", "job_count", "skill_rows", "rate"]
EVIDENCE_INTEGRITY_FIELDS = [
    "run_id",
    "job_id",
    "job_skill_id",
    "canonical",
    "violation_type",
    "details",
]


async def export_matcher_quality_audit(
    run_id: int,
    output_dir: Path,
    *,
    top_skill_seed: int = TOP_SKILL_REVIEW_SEED,
    zero_skill_seed: int = ZERO_SKILL_REVIEW_SEED,
) -> AuditExportResult:
    """Generate deterministic matcher quality audit CSV files for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    async with async_session_factory() as session:
        top_skills = await _top_canonical_skills(session, run_id, 10)
        top_skill_rows = await _top_skill_review_rows(session, run_id, top_skills, top_skill_seed)
        zero_skill_rows = await _zero_skill_review_rows(session, run_id, zero_skill_seed)
        high_skill_rows = await _high_skill_count_rows(session, run_id)
        ambiguous_rows = await _ambiguous_alias_rows(session, run_id)
        frequency_rows = await _frequency_rows(session, run_id)
        integrity_rows = await evidence_integrity_violations(session, run_id)

    _write_csv(output_dir / "top_skill_review.csv", TOP_SKILL_REVIEW_FIELDS, top_skill_rows)
    _write_csv(output_dir / "zero_skill_review.csv", ZERO_SKILL_REVIEW_FIELDS, zero_skill_rows)
    _write_csv(
        output_dir / "high_skill_count_review.csv",
        HIGH_SKILL_COUNT_REVIEW_FIELDS,
        high_skill_rows,
    )
    _write_csv(
        output_dir / "ambiguous_alias_review.csv",
        AMBIGUOUS_ALIAS_REVIEW_FIELDS,
        ambiguous_rows,
    )
    _write_csv(output_dir / "skill_frequency.csv", SKILL_FREQUENCY_FIELDS, frequency_rows)
    _write_csv(
        output_dir / "evidence_integrity_violations.csv",
        EVIDENCE_INTEGRITY_FIELDS,
        integrity_rows,
    )
    return AuditExportResult(
        output_dir=output_dir,
        top_skill_review_rows=len(top_skill_rows),
        zero_skill_review_rows=len(zero_skill_rows),
        high_skill_count_review_rows=len(high_skill_rows),
        ambiguous_alias_review_rows=len(ambiguous_rows),
        skill_frequency_rows=len(frequency_rows),
        evidence_integrity_violations=len(integrity_rows),
    )


async def evidence_integrity_violations(
    session: AsyncSession,
    run_id: int,
) -> list[dict[str, object]]:
    """Validate persisted evidence without mutating it."""
    rows = await session.execute(
        select(JobSkill, Job, Skill)
        .join(Job, Job.id == JobSkill.job_id)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id)
        .order_by(JobSkill.job_id, JobSkill.id)
    )
    violations: list[dict[str, object]] = []
    seen: set[tuple[int, str]] = set()
    for job_skill, job, skill in rows.all():
        canonical = skill.canonical if skill is not None else ""
        if (job_skill.job_id, canonical) in seen:
            violations.append(
                _violation(
                    run_id,
                    job_skill.id,
                    job_skill.job_id,
                    canonical,
                    "duplicate",
                    canonical,
                )
            )
        seen.add((job_skill.job_id, canonical))
        if job is None:
            violations.append(
                _violation(run_id, job_skill.id, job_skill.job_id, canonical, "missing_job", "")
            )
            continue
        if skill is None:
            violations.append(
                _violation(
                    run_id, job_skill.id, job_skill.job_id, canonical, "missing_skill", ""
                )
            )
        if job_skill.evidence_field not in {"title", "description"}:
            violations.append(
                _violation(
                    run_id,
                    job_skill.id,
                    job_skill.job_id,
                    canonical,
                    "invalid_evidence_field",
                    str(job_skill.evidence_field),
                )
            )
            continue
        source_text = job.title if job_skill.evidence_field == "title" else job.description
        if job_skill.start_char < 0 or job_skill.end_char < job_skill.start_char:
            violations.append(
                _violation(run_id, job_skill.id, job_skill.job_id, canonical, "invalid_offsets", "")
            )
            continue
        if job_skill.end_char > len(source_text):
            violations.append(
                _violation(
                    run_id,
                    job_skill.id,
                    job_skill.job_id,
                    canonical,
                    "offset_out_of_range",
                    f"end={job_skill.end_char} len={len(source_text)}",
                )
            )
            continue
        sliced = source_text[job_skill.start_char : job_skill.end_char]
        if sliced != job_skill.evidence_text:
            violations.append(
                _violation(
                    run_id,
                    job_skill.id,
                    job_skill.job_id,
                    canonical,
                    "evidence_text_mismatch",
                    f"slice={sliced!r} evidence={job_skill.evidence_text!r}",
                )
            )
        matched_alias = job_skill.matched_alias or ""
        evidence_text = job_skill.evidence_text or ""
        if normalize_alias(matched_alias) != normalize_alias(evidence_text):
            violations.append(
                _violation(
                    run_id,
                    job_skill.id,
                    job_skill.job_id,
                    canonical,
                    "matched_alias_mismatch",
                    f"alias={matched_alias!r} evidence={evidence_text!r}",
                )
            )
    return violations


def deterministic_sample_ids(ids: list[int], sample_size: int, seed: int) -> list[int]:
    """Return a deterministic random sample from stable sorted IDs."""
    ordered = sorted(ids)
    if len(ordered) <= sample_size:
        return ordered
    sampled = random.Random(seed).sample(ordered, sample_size)
    return sorted(sampled)


def is_ambiguous_alias(alias: str) -> bool:
    """Heuristic for aliases that deserve manual boundary/ambiguity review."""
    folded = normalize_alias(alias)
    punctuation_heavy = bool(re.search(r"[^\w\s]", alias))
    short = len(folded) <= 3
    abbreviation = alias.isupper() and len(alias) <= 5
    common_word = folded in COMMON_AMBIGUOUS_ALIASES
    return short or punctuation_heavy or abbreviation or common_word


def description_excerpt(description: str, limit: int = DESCRIPTION_EXCERPT_CHARS) -> str:
    cleaned = " ".join(description.split())
    return cleaned[:limit]


def context_around(text: str, start: int, end: int, size: int = CONTEXT_CHARS) -> tuple[str, str]:
    return text[max(0, start - size) : start], text[end : min(len(text), end + size)]


async def _top_canonical_skills(
    session: AsyncSession,
    run_id: int,
    limit: int,
) -> list[str]:
    rows = await session.execute(
        select(Skill.canonical, func.count(func.distinct(JobSkill.job_id)).label("job_count"))
        .join(JobSkill, JobSkill.skill_id == Skill.id)
        .where(JobSkill.run_id == run_id)
        .group_by(Skill.canonical)
        .order_by(func.count(func.distinct(JobSkill.job_id)).desc(), Skill.canonical.asc())
        .limit(limit)
    )
    skills = [row.canonical for row in rows.all()]
    if "Artificial Intelligence" not in skills:
        skills = ["Artificial Intelligence", *skills[: max(0, limit - 1)]]
    return skills


async def _top_skill_review_rows(
    session: AsyncSession,
    run_id: int,
    canonical_skills: list[str],
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, canonical in enumerate(canonical_skills):
        records = await _evidence_records_for_skill(session, run_id, canonical)
        sampled_ids = set(
            deterministic_sample_ids([record.job_id for record in records], 10, seed + index)
        )
        for record in records:
            if record.job_id not in sampled_ids:
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "canonical": canonical,
                    "job_id": record.job_id,
                    "matched_alias": record.matched_alias,
                    "evidence_text": record.evidence_text,
                    "evidence_field": record.evidence_field,
                    "start_char": record.start_char,
                    "end_char": record.end_char,
                    "title": record.title,
                    "description_excerpt": description_excerpt(record.description),
                    "source": record.source,
                    "country": record.country,
                    "expected_skills": "",
                    "false_positive": "",
                    "reviewer_notes": "",
                }
            )
    return sorted(rows, key=_canonical_job_sort_key)


def _canonical_job_sort_key(row: dict[str, object]) -> tuple[str, int]:
    job_id = row["job_id"]
    if not isinstance(job_id, int):
        raise TypeError("job_id must be an int")
    return str(row["canonical"]), job_id


async def _evidence_records_for_skill(
    session: AsyncSession,
    run_id: int,
    canonical: str,
) -> list[EvidenceRecord]:
    rows = await session.execute(
        _evidence_record_select(run_id)
        .where(Skill.canonical == canonical)
        .order_by(JobSkill.job_id, JobSkill.id)
    )
    return [_record_from_row(run_id, row) for row in rows.all()]


async def _zero_skill_review_rows(
    session: AsyncSession,
    run_id: int,
    seed: int,
) -> list[dict[str, object]]:
    primary_source = _primary_source_subquery()
    rows = await session.execute(
        select(
            JobEnrichmentResult.job_id,
            Job.title,
            Job.description,
            primary_source.c.source,
            func.coalesce(Job.country, "(unknown)").label("country"),
        )
        .join(Job, Job.id == JobEnrichmentResult.job_id)
        .join(primary_source, primary_source.c.job_id == Job.id)
        .where(
            JobEnrichmentResult.run_id == run_id,
            JobEnrichmentResult.status == "success",
            JobEnrichmentResult.skill_count == 0,
        )
        .order_by(JobEnrichmentResult.job_id)
    )
    records = [
        ZeroSkillRecord(run_id, row.job_id, row.title, row.description, row.source, row.country)
        for row in rows.all()
    ]
    sampled_ids = set(deterministic_sample_ids([record.job_id for record in records], 50, seed))
    return [
        {
            "run_id": run_id,
            "job_id": record.job_id,
            "title": record.title,
            "description_excerpt": description_excerpt(record.description),
            "source": record.source,
            "country": record.country,
            "expected_skills": "",
            "missed_skill": "",
            "ontology_gap": "",
            "reviewer_notes": "",
        }
        for record in records
        if record.job_id in sampled_ids
    ]


async def _high_skill_count_rows(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    primary_source = _primary_source_subquery()
    rows = await session.execute(
        select(
            JobEnrichmentResult.job_id,
            JobEnrichmentResult.skill_count,
            Job.title,
            Job.description,
            primary_source.c.source,
            func.coalesce(Job.country, "(unknown)").label("country"),
        )
        .join(Job, Job.id == JobEnrichmentResult.job_id)
        .join(primary_source, primary_source.c.job_id == Job.id)
        .where(JobEnrichmentResult.run_id == run_id, JobEnrichmentResult.status == "success")
        .order_by(JobEnrichmentResult.skill_count.desc(), JobEnrichmentResult.job_id.asc())
        .limit(25)
    )
    result: list[dict[str, object]] = []
    for row in rows.all():
        skills = await _skills_for_job(session, run_id, row.job_id)
        result.append(
            {
                "run_id": run_id,
                "job_id": row.job_id,
                "skill_count": int(row.skill_count),
                "title": row.title,
                "description_excerpt": description_excerpt(row.description),
                "source": row.source,
                "country": row.country,
                "detected_skills": "; ".join(skill.canonical for skill in skills),
                "evidence": " | ".join(
                    f"{skill.canonical}={skill.evidence_text!r}@{skill.evidence_field}:"
                    f"{skill.start_char}-{skill.end_char}"
                    for skill in skills
                ),
                "expected_skills": "",
                "false_positive_skills": "",
                "reviewer_notes": "",
            }
        )
    return result


async def _ambiguous_alias_rows(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    rows = await session.execute(
        _evidence_record_select(run_id).order_by(JobSkill.job_id, JobSkill.id)
    )
    candidates = [
        _record_from_row(run_id, row)
        for row in rows.all()
        if is_ambiguous_alias(row.matched_alias or "")
    ]
    output: list[dict[str, object]] = []
    for record in candidates[:100]:
        source_text = record.title if record.evidence_field == "title" else record.description
        before, after = context_around(source_text, record.start_char, record.end_char)
        output.append(
            {
                "run_id": run_id,
                "job_id": record.job_id,
                "canonical": record.canonical,
                "matched_alias": record.matched_alias,
                "evidence_text": record.evidence_text,
                "evidence_field": record.evidence_field,
                "start_char": record.start_char,
                "end_char": record.end_char,
                "context_before": before,
                "context_after": after,
                "title": record.title,
                "source": record.source,
                "country": record.country,
                "expected_skills": "",
                "false_positive": "",
                "reviewer_notes": "",
            }
        )
    return output


@dataclass(frozen=True)
class _JobSkillSummary:
    canonical: str
    evidence_text: str
    evidence_field: str
    start_char: int
    end_char: int


async def _skills_for_job(
    session: AsyncSession,
    run_id: int,
    job_id: int,
) -> list[_JobSkillSummary]:
    rows = await session.execute(
        select(
            Skill.canonical,
            JobSkill.evidence_text,
            JobSkill.evidence_field,
            JobSkill.start_char,
            JobSkill.end_char,
        )
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id, JobSkill.job_id == job_id)
        .order_by(Skill.canonical)
    )
    return [
        _JobSkillSummary(
            canonical=row.canonical,
            evidence_text=row.evidence_text or "",
            evidence_field=row.evidence_field,
            start_char=int(row.start_char),
            end_char=int(row.end_char),
        )
        for row in rows.all()
    ]


async def _frequency_rows(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    output.extend(await _top_skill_frequency_rows(session, run_id))
    output.extend(await _top_alias_frequency_rows(session, run_id))
    output.extend(await _skill_frequency_by_source(session, run_id))
    output.extend(await _skill_frequency_by_country(session, run_id))
    output.extend(await _zero_skill_rate_by_source(session, run_id))
    output.extend(await _zero_skill_rate_by_country(session, run_id))
    output.extend(await _skill_count_distribution(session, run_id))
    output.extend(await _unusually_high_skill_jobs(session, run_id))
    return output


async def _top_skill_frequency_rows(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    rows = await session.execute(
        select(
            Skill.canonical,
            func.count(func.distinct(JobSkill.job_id)).label("job_count"),
            func.count(JobSkill.id).label("skill_rows"),
        )
        .join(JobSkill, JobSkill.skill_id == Skill.id)
        .where(JobSkill.run_id == run_id)
        .group_by(Skill.canonical)
        .order_by(func.count(func.distinct(JobSkill.job_id)).desc(), Skill.canonical.asc())
        .limit(30)
    )
    return [
        _frequency_row("top_canonical_skill", row.canonical, "", row.job_count, row.skill_rows)
        for row in rows.all()
    ]


async def _top_alias_frequency_rows(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    rows = await session.execute(
        select(
            JobSkill.matched_alias,
            Skill.canonical,
            func.count(func.distinct(JobSkill.job_id)).label("job_count"),
            func.count(JobSkill.id).label("skill_rows"),
        )
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id)
        .group_by(JobSkill.matched_alias, Skill.canonical)
        .order_by(
            func.count(func.distinct(JobSkill.job_id)).desc(),
            JobSkill.matched_alias.asc(),
            Skill.canonical.asc(),
        )
        .limit(30)
    )
    return [
        _frequency_row(
            "top_matched_alias",
            row.matched_alias or "",
            row.canonical,
            row.job_count,
            row.skill_rows,
        )
        for row in rows.all()
    ]


async def _skill_frequency_by_source(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    rows = await session.execute(
        select(
            RawJob.source,
            Skill.canonical,
            func.count(func.distinct(JobSkill.job_id)).label("job_count"),
            func.count(JobSkill.id).label("skill_rows"),
        )
        .join(Job, Job.id == JobSkill.job_id)
        .join(JobSource, JobSource.job_id == Job.id)
        .join(RawJob, RawJob.id == JobSource.raw_job_id)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id)
        .group_by(RawJob.source, Skill.canonical)
        .order_by(
            RawJob.source.asc(),
            func.count(func.distinct(JobSkill.job_id)).desc(),
            Skill.canonical.asc(),
        )
        .limit(30)
    )
    return [
        _frequency_row("skill_by_source", row.source, row.canonical, row.job_count, row.skill_rows)
        for row in rows.all()
    ]


async def _skill_frequency_by_country(
    session: AsyncSession,
    run_id: int,
) -> list[dict[str, object]]:
    rows = await session.execute(
        select(
            func.coalesce(Job.country, "(unknown)").label("country"),
            Skill.canonical,
            func.count(func.distinct(JobSkill.job_id)).label("job_count"),
            func.count(JobSkill.id).label("skill_rows"),
        )
        .join(Job, Job.id == JobSkill.job_id)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .where(JobSkill.run_id == run_id)
        .group_by("country", Skill.canonical)
        .order_by(
            "country",
            func.count(func.distinct(JobSkill.job_id)).desc(),
            Skill.canonical.asc(),
        )
        .limit(30)
    )
    return [
        _frequency_row(
            "skill_by_country", row.country, row.canonical, row.job_count, row.skill_rows
        )
        for row in rows.all()
    ]


async def _zero_skill_rate_by_source(
    session: AsyncSession,
    run_id: int,
) -> list[dict[str, object]]:
    primary_source = _primary_source_subquery()
    rows = await session.execute(
        select(primary_source.c.source, JobEnrichmentResult.skill_count)
        .join(Job, Job.id == JobEnrichmentResult.job_id)
        .join(primary_source, primary_source.c.job_id == Job.id)
        .where(JobEnrichmentResult.run_id == run_id, JobEnrichmentResult.status == "success")
        .order_by(primary_source.c.source, JobEnrichmentResult.job_id)
    )
    return _zero_rate_rows(
        "zero_skill_rate_by_source",
        [(row.source, int(row.skill_count)) for row in rows.all()],
    )


async def _zero_skill_rate_by_country(
    session: AsyncSession,
    run_id: int,
) -> list[dict[str, object]]:
    rows = await session.execute(
        select(
            func.coalesce(Job.country, "(unknown)").label("country"),
            JobEnrichmentResult.skill_count,
        )
        .join(Job, Job.id == JobEnrichmentResult.job_id)
        .where(JobEnrichmentResult.run_id == run_id, JobEnrichmentResult.status == "success")
        .order_by("country", JobEnrichmentResult.job_id)
    )
    return _zero_rate_rows(
        "zero_skill_rate_by_country",
        [(row.country, int(row.skill_count)) for row in rows.all()],
    )


async def _skill_count_distribution(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    rows = await session.execute(
        select(
            JobEnrichmentResult.skill_count,
            func.count(JobEnrichmentResult.job_id).label("jobs"),
        )
        .where(JobEnrichmentResult.run_id == run_id, JobEnrichmentResult.status == "success")
        .group_by(JobEnrichmentResult.skill_count)
        .order_by(JobEnrichmentResult.skill_count.asc())
    )
    return [
        _frequency_row("skill_count_distribution", str(row.skill_count), "", row.jobs, 0)
        for row in rows.all()
    ]


async def _unusually_high_skill_jobs(session: AsyncSession, run_id: int) -> list[dict[str, object]]:
    rows = await session.execute(
        select(JobEnrichmentResult.job_id, JobEnrichmentResult.skill_count)
        .where(JobEnrichmentResult.run_id == run_id, JobEnrichmentResult.status == "success")
        .order_by(JobEnrichmentResult.skill_count.desc(), JobEnrichmentResult.job_id.asc())
        .limit(25)
    )
    return [
        _frequency_row("unusually_high_skill_job", str(row.job_id), "", row.skill_count, 0)
        for row in rows.all()
    ]


def _zero_rate_rows(metric: str, rows: list[tuple[str, int]]) -> list[dict[str, object]]:
    totals: Counter[str] = Counter()
    zeros: Counter[str] = Counter()
    for key, skill_count in rows:
        totals[key] += 1
        if skill_count == 0:
            zeros[key] += 1
    return [
        _frequency_row(metric, key, "", zeros[key], 0, zeros[key] / totals[key])
        for key in sorted(totals)
    ]


def _frequency_row(
    metric: str,
    key: str,
    secondary_key: str,
    job_count: int,
    skill_rows: int,
    rate: float = 0.0,
) -> dict[str, object]:
    return {
        "metric": metric,
        "key": key,
        "secondary_key": secondary_key,
        "job_count": int(job_count),
        "skill_rows": int(skill_rows),
        "rate": rate,
    }


def _evidence_record_select(run_id: int) -> Any:
    primary_source = _primary_source_subquery()
    return (
        select(
            JobSkill.job_id,
            Skill.canonical,
            JobSkill.matched_alias,
            JobSkill.evidence_text,
            JobSkill.evidence_field,
            JobSkill.start_char,
            JobSkill.end_char,
            Job.title,
            Job.description,
            primary_source.c.source,
            func.coalesce(Job.country, "(unknown)").label("country"),
        )
        .join(Job, Job.id == JobSkill.job_id)
        .join(Skill, Skill.id == JobSkill.skill_id)
        .join(primary_source, primary_source.c.job_id == Job.id)
        .where(JobSkill.run_id == run_id)
    )


def _primary_source_subquery() -> Any:
    return (
        select(JobSource.job_id, func.min(RawJob.source).label("source"))
        .join(RawJob, RawJob.id == JobSource.raw_job_id)
        .group_by(JobSource.job_id)
        .subquery("primary_job_source")
    )


def _record_from_row(run_id: int, row: Any) -> EvidenceRecord:
    return EvidenceRecord(
        run_id=run_id,
        job_id=int(row.job_id),
        canonical=row.canonical,
        matched_alias=row.matched_alias or "",
        evidence_text=row.evidence_text or "",
        evidence_field=row.evidence_field,
        start_char=int(row.start_char),
        end_char=int(row.end_char),
        title=row.title,
        description=row.description,
        source=row.source,
        country=row.country,
    )


def _violation(
    run_id: int,
    job_skill_id: int,
    job_id: int,
    canonical: str,
    violation_type: str,
    details: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "job_id": job_id,
        "job_skill_id": job_skill_id,
        "canonical": canonical,
        "violation_type": violation_type,
        "details": details,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dataclass_field_names(cls: type[Any]) -> list[str]:
    return [field.name for field in fields(cls)]
