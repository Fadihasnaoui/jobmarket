"""Sync the skills ontology YAML into the `skills` table. Idempotent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.db.models import Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.ontology import SkillEntry, SkillsOntology, load_ontology

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class _SkillSyncComparison:
    database_count: int
    missing: list[str]
    extra: list[str]
    duplicates: list[str]


class SkillSyncInspection(TypedDict):
    """Pre-sync comparison of ontology and database skills."""

    ontology_count: int
    database_count_before: int
    missing_before: list[str]
    extra_before: list[str]
    duplicate_canonicals_before: list[str]


class SkillSyncFinalInspection(TypedDict):
    """Post-sync comparison of ontology and database skills."""

    database_count_after: int
    final_missing: list[str]
    final_extra: list[str]
    final_duplicate_canonicals: list[str]
    final_synced: bool


class SkillsSyncReport(SkillSyncInspection, SkillSyncFinalInspection):
    """Observable result of ontology/database skill synchronization."""

    inserted: int
    updated: int
    synced: int


async def sync_skills(ontology: SkillsOntology | None = None) -> SkillsSyncReport:
    """Synchronize ontology skills into the database without deleting existing rows.

    Missing canonical skills are inserted. Existing rows keep their primary keys and are
    updated only when category or parent metadata differs from the ontology. Extra database
    skills are reported but not deleted automatically.
    """
    onto = ontology or load_ontology()

    async with async_session_factory() as session:
        before = await inspect_skill_sync(session, onto)
        inserted = await _insert_missing_entries(session, onto, set(before["missing_before"]))
        await session.flush()
        updated = await _update_changed_metadata(session, onto)
        await session.commit()

        final = await inspect_final_skill_sync(session, onto)

    report: SkillsSyncReport = {
        "ontology_count": before["ontology_count"],
        "database_count_before": before["database_count_before"],
        "database_count_after": final["database_count_after"],
        "missing_before": before["missing_before"],
        "extra_before": before["extra_before"],
        "duplicate_canonicals_before": before["duplicate_canonicals_before"],
        "inserted": inserted,
        "updated": updated,
        "synced": len(onto),
        "final_missing": final["final_missing"],
        "final_extra": final["final_extra"],
        "final_duplicate_canonicals": final["final_duplicate_canonicals"],
        "final_synced": final["final_synced"],
    }
    logger.info(
        "skills_sync_complete",
        ontology_count=report["ontology_count"],
        inserted=inserted,
        updated=updated,
        final_synced=report["final_synced"],
    )
    return report


async def inspect_skill_sync(
    session: AsyncSession,
    ontology: SkillsOntology,
) -> SkillSyncInspection:
    """Compare ontology canonical skills with the database `skills` table before sync."""
    comparison = await _compare_skill_sync(session, ontology)
    return {
        "ontology_count": len(ontology),
        "database_count_before": comparison.database_count,
        "missing_before": comparison.missing,
        "extra_before": comparison.extra,
        "duplicate_canonicals_before": comparison.duplicates,
    }


async def inspect_final_skill_sync(
    session: AsyncSession,
    ontology: SkillsOntology,
) -> SkillSyncFinalInspection:
    """Compare ontology canonical skills with the database `skills` table after sync."""
    comparison = await _compare_skill_sync(session, ontology)
    return {
        "database_count_after": comparison.database_count,
        "final_missing": comparison.missing,
        "final_extra": comparison.extra,
        "final_duplicate_canonicals": comparison.duplicates,
        "final_synced": not comparison.missing and not comparison.duplicates,
    }


async def _compare_skill_sync(
    session: AsyncSession,
    ontology: SkillsOntology,
) -> _SkillSyncComparison:
    ontology_canonicals = {entry.canonical for entry in ontology.entries}
    db_rows = list(await session.scalars(select(Skill.canonical)))
    db_canonicals = set(db_rows)
    return _SkillSyncComparison(
        database_count=len(db_rows),
        missing=sorted(ontology_canonicals - db_canonicals),
        extra=sorted(db_canonicals - ontology_canonicals),
        duplicates=await _duplicate_canonicals(session),
    )


async def _insert_missing_entries(
    session: AsyncSession,
    ontology: SkillsOntology,
    missing_canonicals: set[str],
) -> int:
    entries = [entry for entry in ontology.entries if entry.canonical in missing_canonicals]
    session.add_all(
        [Skill(canonical=entry.canonical, category=entry.category) for entry in entries]
    )
    return len(entries)


async def _update_changed_metadata(session: AsyncSession, ontology: SkillsOntology) -> int:
    rows = list(await session.scalars(select(Skill).where(Skill.canonical.is_not(None))))
    by_canonical = {row.canonical: row for row in rows}
    updated_ids: set[int] = set()

    for entry in ontology.entries:
        row = by_canonical[entry.canonical]
        if row.category != entry.category:
            row.category = entry.category
            updated_ids.add(row.id)

    await session.flush()
    rows = list(await session.scalars(select(Skill).where(Skill.canonical.is_not(None))))
    by_canonical = {row.canonical: row for row in rows}
    for entry in ontology.entries:
        row = by_canonical[entry.canonical]
        expected_parent_id = _expected_parent_id(entry, by_canonical)
        if row.parent_id != expected_parent_id:
            row.parent_id = expected_parent_id
            updated_ids.add(row.id)

    return len(updated_ids)


def _expected_parent_id(entry: SkillEntry, by_canonical: dict[str, Skill]) -> int | None:
    if entry.parent is None:
        return None
    return by_canonical[entry.parent].id


async def _duplicate_canonicals(session: AsyncSession) -> list[str]:
    rows = await session.execute(
        select(Skill.canonical)
        .group_by(Skill.canonical)
        .having(func.count(Skill.id) > 1)
        .order_by(Skill.canonical)
    )
    return list(rows.scalars().all())
