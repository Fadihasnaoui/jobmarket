"""Deterministic skill enrichment runner for normalized jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import structlog
from sqlalchemy import Select, delete, exists, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import (
    EnrichmentFailure,
    EnrichmentRun,
    Job,
    JobEnrichmentResult,
    JobSkill,
    JobSource,
    RawJob,
    Skill,
)
from jobmarket.db.session import async_session_factory
from jobmarket.skills.matcher import SkillMatch, find_skills, get_matcher_versions
from jobmarket.skills.ontology import SkillsOntology, load_ontology

logger = structlog.get_logger(__name__)

METHOD = "matcher"
RUN_TYPE = "matcher"
_FAILURE_STAGE = "matcher"
_SUCCESS_STATUS = "success"
_FAILED_STATUS = "failed"
_RUNNING_STATUS = "running"
_PARTIAL_STATUS = "partial"

SkillMatcher = Callable[[str, SkillsOntology], list[SkillMatch]]


class EnrichmentConfigError(ValueError):
    """Raised when a requested enrichment run cannot be configured safely."""


class OntologySyncError(RuntimeError):
    """Raised when the ontology and skills table do not describe the same skills."""


@dataclass(frozen=True)
class MatcherEnrichmentConfig:
    """Selection and batching options for deterministic matcher enrichment."""

    source: str | None = None
    country: str | None = None
    min_job_id: int | None = None
    max_job_id: int | None = None
    limit: int | None = None
    batch_size: int = 500

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise EnrichmentConfigError("batch_size must be at least 1")
        if self.limit is not None and self.limit < 1:
            raise EnrichmentConfigError("limit must be at least 1 when provided")
        if self.min_job_id is not None and self.min_job_id < 1:
            raise EnrichmentConfigError("min_job_id must be positive when provided")
        if self.max_job_id is not None and self.max_job_id < 1:
            raise EnrichmentConfigError("max_job_id must be positive when provided")
        if (
            self.min_job_id is not None
            and self.max_job_id is not None
            and self.min_job_id > self.max_job_id
        ):
            raise EnrichmentConfigError("min_job_id cannot be greater than max_job_id")

    def to_run_config(self) -> dict[str, int | str | None]:
        """JSON-stable config persisted on the enrichment run."""
        return asdict(self)


@dataclass(frozen=True)
class MatcherEnrichmentSummary:
    """Final observable state for a matcher enrichment execution."""

    run_id: int
    status: str
    jobs_total: int
    jobs_processed: int
    jobs_succeeded: int
    jobs_failed: int
    jobs_with_skills: int
    zero_skill_jobs: int


@dataclass(frozen=True)
class _JobRow:
    id: int
    title: str
    description: str


@dataclass(frozen=True)
class _FieldEvidence:
    canonical: str
    matched_alias: str
    evidence_text: str
    evidence_field: str
    start_char: int
    end_char: int


async def run_matcher_enrichment(
    config: MatcherEnrichmentConfig | None = None,
    *,
    resume_run_id: int | None = None,
    force: bool = False,
    match_func: SkillMatcher = find_skills,
) -> MatcherEnrichmentSummary:
    """Run deterministic skill enrichment and persist per-job results.

    Force mode always creates a fresh run and preserves historical data. Resume mode reuses
    the requested run, verifies versions/config, and only processes jobs without a successful
    ledger row in that same run.
    """
    if resume_run_id is not None and force:
        raise EnrichmentConfigError("resume_run_id and force cannot be used together")

    run_config = config or MatcherEnrichmentConfig()
    ontology = load_ontology()
    versions = get_matcher_versions(ontology)

    async with async_session_factory() as session:
        skill_ids = await _load_skill_ids(session, ontology)
        run = await _prepare_run(
            session,
            config=run_config,
            matcher_version=versions.matcher_version,
            ontology_version=versions.ontology_version,
            resume_run_id=resume_run_id,
        )
        await session.commit()
        run_id = run.id

    try:
        async with async_session_factory() as session:
            await _set_run_running(session, run_id)
            jobs_total = await _count_run_jobs(
                session,
                run_id=run_id,
                config=run_config,
                matcher_version=versions.matcher_version,
                ontology_version=versions.ontology_version,
                skip_previous_successes=resume_run_id is None and not force,
            )
            await _refresh_run_counters(session, run_id, jobs_total)
            await session.commit()

        last_job_id = 0
        while True:
            async with async_session_factory() as session:
                jobs = await _fetch_job_batch(
                    session,
                    run_id=run_id,
                    config=run_config,
                    batch_size=run_config.batch_size,
                    last_job_id=last_job_id,
                    matcher_version=versions.matcher_version,
                    ontology_version=versions.ontology_version,
                    skip_previous_successes=resume_run_id is None and not force,
                )
                if not jobs:
                    break

                for job in jobs:
                    last_job_id = job.id
                    await _process_job(
                        session,
                        run_id=run_id,
                        job=job,
                        ontology=ontology,
                        skill_ids=skill_ids,
                        match_func=match_func,
                    )

                await _refresh_run_counters(
                    session,
                    run_id,
                    await _count_run_jobs(
                        session,
                        run_id=run_id,
                        config=run_config,
                        matcher_version=versions.matcher_version,
                        ontology_version=versions.ontology_version,
                        skip_previous_successes=resume_run_id is None and not force,
                    ),
                )
                await session.commit()

        async with async_session_factory() as session:
            summary = await _finish_run(session, run_id)
            await session.commit()
            logger.info("matcher_enrichment_complete", **asdict(summary))
            return summary
    except Exception as exc:
        async with async_session_factory() as session:
            await _mark_run_failed(session, run_id, str(exc))
            await session.commit()
        raise


async def _load_skill_ids(session: AsyncSession, ontology: SkillsOntology) -> dict[str, int]:
    rows = await session.execute(select(Skill.canonical, Skill.id))
    db_skill_ids = {canonical: skill_id for canonical, skill_id in rows.all()}
    ontology_canonicals = {entry.canonical for entry in ontology.entries}
    db_canonicals = set(db_skill_ids)

    missing = ontology_canonicals - db_canonicals
    extra = db_canonicals - ontology_canonicals
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={sorted(missing)[:5]}")
        if extra:
            details.append(f"extra={sorted(extra)[:5]}")
        message = "skills table is not synchronized with ontology: " + ", ".join(details)
        raise OntologySyncError(message)
    return db_skill_ids


async def _prepare_run(
    session: AsyncSession,
    *,
    config: MatcherEnrichmentConfig,
    matcher_version: str,
    ontology_version: str,
    resume_run_id: int | None,
) -> EnrichmentRun:
    if resume_run_id is not None:
        run = await session.get(EnrichmentRun, resume_run_id)
        if run is None:
            raise EnrichmentConfigError(f"enrichment run {resume_run_id} does not exist")
        if run.run_type != RUN_TYPE:
            raise EnrichmentConfigError(f"run {resume_run_id} is not a matcher run")
        if run.matcher_version != matcher_version or run.ontology_version != ontology_version:
            raise EnrichmentConfigError("resume run versions do not match current matcher/ontology")
        if run.config != config.to_run_config():
            raise EnrichmentConfigError("resume run config does not match requested config")
        run.status = _RUNNING_STATUS
        run.finished_at = None
        run.error = None
        run.updated_at = datetime.now(UTC)
        return run

    run = EnrichmentRun(
        run_type=RUN_TYPE,
        status=_RUNNING_STATUS,
        matcher_version=matcher_version,
        ontology_version=ontology_version,
        config=config.to_run_config(),
        jobs_total=0,
        jobs_processed=0,
        jobs_succeeded=0,
        jobs_failed=0,
        jobs_with_skills=0,
        zero_skill_jobs=0,
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(run)
    await session.flush()
    return run


async def _set_run_running(session: AsyncSession, run_id: int) -> None:
    now = datetime.now(UTC)
    await session.execute(
        update(EnrichmentRun)
        .where(EnrichmentRun.id == run_id)
        .values(status=_RUNNING_STATUS, finished_at=None, error=None, updated_at=now)
    )


async def _process_job(
    session: AsyncSession,
    *,
    run_id: int,
    job: _JobRow,
    ontology: SkillsOntology,
    skill_ids: dict[str, int],
    match_func: SkillMatcher,
) -> None:
    try:
        await session.execute(
            delete(JobSkill).where(
                JobSkill.run_id == run_id,
                JobSkill.job_id == job.id,
                JobSkill.method == METHOD,
            )
        )
        evidence = _match_job(job, ontology, match_func)
        for item in evidence:
            await _upsert_job_skill(session, run_id, job.id, skill_ids[item.canonical], item)
        await _upsert_job_result(session, run_id, job.id, _SUCCESS_STATUS, len(evidence))
    except Exception as exc:
        await _upsert_job_result(session, run_id, job.id, _FAILED_STATUS, 0)
        await _upsert_failure(session, run_id, job.id, _FAILURE_STAGE, str(exc))
        logger.warning(
            "matcher_enrichment_job_failed",
            run_id=run_id,
            job_id=job.id,
            error=str(exc),
        )


def _match_job(
    job: _JobRow,
    ontology: SkillsOntology,
    match_func: SkillMatcher,
) -> list[_FieldEvidence]:
    title = job.title or ""
    description = job.description or ""
    combined = f"{title}\n\n{description}"
    description_offset = len(title) + 2
    evidence: list[_FieldEvidence] = []

    for match in match_func(combined, ontology):
        field_item = _field_evidence_from_match(match, title, description, description_offset)
        _verify_evidence(field_item, title, description)
        evidence.append(field_item)

    return evidence


def _field_evidence_from_match(
    match: SkillMatch,
    title: str,
    description: str,
    description_offset: int,
) -> _FieldEvidence:
    if match.end <= len(title):
        start = match.start
        end = match.end
        field = "title"
        source_text = title
    elif match.start >= description_offset:
        start = match.start - description_offset
        end = match.end - description_offset
        field = "description"
        source_text = description
    else:
        raise ValueError(f"matched skill {match.canonical!r} crosses title/description boundary")

    return _FieldEvidence(
        canonical=match.canonical,
        matched_alias=match.alias,
        evidence_text=source_text[start:end],
        evidence_field=field,
        start_char=start,
        end_char=end,
    )


def _verify_evidence(evidence: _FieldEvidence, title: str, description: str) -> None:
    original = title if evidence.evidence_field == "title" else description
    if original[evidence.start_char : evidence.end_char] != evidence.evidence_text:
        raise ValueError(f"evidence offsets do not match original {evidence.evidence_field} text")


async def _upsert_job_skill(
    session: AsyncSession,
    run_id: int,
    job_id: int,
    skill_id: int,
    evidence: _FieldEvidence,
) -> None:
    stmt = (
        insert(JobSkill)
        .values(
            run_id=run_id,
            job_id=job_id,
            skill_id=skill_id,
            method=METHOD,
            matched_alias=evidence.matched_alias,
            evidence_text=evidence.evidence_text,
            evidence_field=evidence.evidence_field,
            start_char=evidence.start_char,
            end_char=evidence.end_char,
            confidence=None,
        )
        .on_conflict_do_update(
            constraint="uq_job_skills_run_job_skill_method",
            set_={
                "matched_alias": evidence.matched_alias,
                "evidence_text": evidence.evidence_text,
                "evidence_field": evidence.evidence_field,
                "start_char": evidence.start_char,
                "end_char": evidence.end_char,
                "confidence": None,
            },
        )
    )
    await session.execute(stmt)


async def _upsert_job_result(
    session: AsyncSession,
    run_id: int,
    job_id: int,
    status: str,
    skill_count: int,
) -> None:
    now = datetime.now(UTC)
    stmt = (
        insert(JobEnrichmentResult)
        .values(
            run_id=run_id,
            job_id=job_id,
            status=status,
            skill_count=skill_count,
            processed_at=now,
            updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_job_enrichment_results_run_job",
            set_={
                "status": status,
                "skill_count": skill_count,
                "processed_at": now,
                "updated_at": now,
            },
        )
    )
    await session.execute(stmt)


async def _upsert_failure(
    session: AsyncSession, run_id: int, job_id: int, stage: str, error: str) -> None:
    now = datetime.now(UTC)
    stmt = (
        insert(EnrichmentFailure)
        .values(
            run_id=run_id,
            job_id=job_id,
            stage=stage,
            attempts=1,
            error=error,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_enrichment_failures_run_job_stage",
            set_={
                "attempts": EnrichmentFailure.attempts + 1,
                "error": error,
                "updated_at": now,
            },
        )
    )
    await session.execute(stmt)


async def _finish_run(session: AsyncSession, run_id: int) -> MatcherEnrichmentSummary:
    total = await _scalar_int(
        session,
        select(EnrichmentRun.jobs_total).where(EnrichmentRun.id == run_id),
    )
    succeeded, failed, with_skills, zero_skill = await _result_counts(session, run_id)
    processed = succeeded + failed
    if total == 0 or (failed == 0 and processed >= total):
        status = _SUCCESS_STATUS
    elif succeeded > 0:
        status = _PARTIAL_STATUS
    else:
        status = _FAILED_STATUS

    now = datetime.now(UTC)
    await session.execute(
        update(EnrichmentRun)
        .where(EnrichmentRun.id == run_id)
        .values(
            status=status,
            jobs_processed=processed,
            jobs_succeeded=succeeded,
            jobs_failed=failed,
            jobs_with_skills=with_skills,
            zero_skill_jobs=zero_skill,
            finished_at=now,
            updated_at=now,
            error=None if status != _FAILED_STATUS else "all selected jobs failed",
        )
    )
    return MatcherEnrichmentSummary(
        run_id=run_id,
        status=status,
        jobs_total=total,
        jobs_processed=processed,
        jobs_succeeded=succeeded,
        jobs_failed=failed,
        jobs_with_skills=with_skills,
        zero_skill_jobs=zero_skill,
    )


async def _mark_run_failed(session: AsyncSession, run_id: int, error: str) -> None:
    succeeded, failed, with_skills, zero_skill = await _result_counts(session, run_id)
    processed = succeeded + failed
    now = datetime.now(UTC)
    await session.execute(
        update(EnrichmentRun)
        .where(EnrichmentRun.id == run_id)
        .values(
            status=_FAILED_STATUS,
            jobs_processed=processed,
            jobs_succeeded=succeeded,
            jobs_failed=failed,
            jobs_with_skills=with_skills,
            zero_skill_jobs=zero_skill,
            finished_at=now,
            updated_at=now,
            error=error,
        )
    )


async def _refresh_run_counters(session: AsyncSession, run_id: int, jobs_total: int) -> None:
    succeeded, failed, with_skills, zero_skill = await _result_counts(session, run_id)
    now = datetime.now(UTC)
    await session.execute(
        update(EnrichmentRun)
        .where(EnrichmentRun.id == run_id)
        .values(
            jobs_total=jobs_total,
            jobs_processed=succeeded + failed,
            jobs_succeeded=succeeded,
            jobs_failed=failed,
            jobs_with_skills=with_skills,
            zero_skill_jobs=zero_skill,
            updated_at=now,
        )
    )


async def _result_counts(session: AsyncSession, run_id: int) -> tuple[int, int, int, int]:
    base = select(func.count()).select_from(JobEnrichmentResult).where(
        JobEnrichmentResult.run_id == run_id
    )
    succeeded = await _scalar_int(
        session,
        base.where(JobEnrichmentResult.status == _SUCCESS_STATUS),
    )
    failed = await _scalar_int(
        session,
        base.where(JobEnrichmentResult.status == _FAILED_STATUS),
    )
    with_skills = await _scalar_int(
        session,
        base.where(
            JobEnrichmentResult.status == _SUCCESS_STATUS,
            JobEnrichmentResult.skill_count > 0,
        ),
    )
    zero_skill = await _scalar_int(
        session,
        base.where(
            JobEnrichmentResult.status == _SUCCESS_STATUS,
            JobEnrichmentResult.skill_count == 0,
        ),
    )
    return succeeded, failed, with_skills, zero_skill


async def _count_run_jobs(
    session: AsyncSession,
    *,
    run_id: int,
    config: MatcherEnrichmentConfig,
    matcher_version: str,
    ontology_version: str,
    skip_previous_successes: bool,
) -> int:
    base = _selected_jobs_subquery(config)
    stmt = select(func.count()).select_from(base)
    if skip_previous_successes:
        stmt = stmt.where(
            ~exists()
            .where(
                JobEnrichmentResult.job_id == base.c.id,
                JobEnrichmentResult.status == _SUCCESS_STATUS,
                JobEnrichmentResult.run_id == EnrichmentRun.id,
                EnrichmentRun.id != run_id,
                EnrichmentRun.run_type == RUN_TYPE,
                EnrichmentRun.matcher_version == matcher_version,
                EnrichmentRun.ontology_version == ontology_version,
            )
            .correlate(base)
        )
    return await _scalar_int(session, stmt)


async def _fetch_job_batch(
    session: AsyncSession,
    *,
    run_id: int,
    config: MatcherEnrichmentConfig,
    batch_size: int,
    last_job_id: int,
    matcher_version: str,
    ontology_version: str,
    skip_previous_successes: bool,
) -> list[_JobRow]:
    base = _selected_jobs_subquery(config)
    stmt = (
        select(base.c.id, base.c.title, base.c.description)
        .where(base.c.id > last_job_id)
        .order_by(base.c.id)
        .limit(batch_size)
    )
    stmt = stmt.where(
        ~exists().where(
            JobEnrichmentResult.run_id == run_id,
            JobEnrichmentResult.job_id == base.c.id,
            JobEnrichmentResult.status == _SUCCESS_STATUS,
        )
    )
    if skip_previous_successes:
        stmt = stmt.where(
            ~exists()
            .where(
                JobEnrichmentResult.job_id == base.c.id,
                JobEnrichmentResult.status == _SUCCESS_STATUS,
                JobEnrichmentResult.run_id == EnrichmentRun.id,
                EnrichmentRun.id != run_id,
                EnrichmentRun.run_type == RUN_TYPE,
                EnrichmentRun.matcher_version == matcher_version,
                EnrichmentRun.ontology_version == ontology_version,
            )
            .correlate(base)
        )

    rows = await session.execute(stmt)
    return [_JobRow(id=row.id, title=row.title, description=row.description) for row in rows.all()]


def _selected_jobs_subquery(config: MatcherEnrichmentConfig) -> Any:
    stmt = _base_job_select(config).order_by(Job.id)
    if config.limit is not None:
        stmt = stmt.limit(config.limit)
    return stmt.subquery("selected_jobs")


def _base_job_select(config: MatcherEnrichmentConfig) -> Select[tuple[int, str, str]]:
    stmt = select(Job.id, Job.title, Job.description)
    if config.country is not None:
        stmt = stmt.where(Job.country == config.country)
    if config.min_job_id is not None:
        stmt = stmt.where(Job.id >= config.min_job_id)
    if config.max_job_id is not None:
        stmt = stmt.where(Job.id <= config.max_job_id)
    if config.source is not None:
        stmt = stmt.where(
            exists()
            .where(
                JobSource.job_id == Job.id,
                JobSource.raw_job_id == RawJob.id,
                RawJob.source == config.source,
            )
            .correlate(Job)
        )
    return stmt


async def _scalar_int(session: AsyncSession, stmt: Any) -> int:
    value = await session.scalar(stmt)
    return int(value or 0)
