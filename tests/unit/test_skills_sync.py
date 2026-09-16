"""Unit tests for ontology-to-database skill synchronization."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import delete, func, select

from jobmarket.db.models import Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.enrichment import _load_skill_ids
from jobmarket.skills.ontology import SkillEntry, SkillsOntology, load_ontology
from jobmarket.skills.sync import inspect_skill_sync, sync_skills


async def test_sync_inserts_missing_skills_preserves_ids_and_links_parent() -> None:
    suffix = uuid4().hex[:8]
    parent_name = f"pytest-sync-parent-{suffix}"
    child_name = f"pytest-sync-child-{suffix}"
    ontology = SkillsOntology(
        [
            SkillEntry(canonical=parent_name, category="language"),
            SkillEntry(canonical=child_name, category="framework", parent=parent_name),
        ]
    )

    try:
        async with async_session_factory() as session:
            parent = Skill(canonical=parent_name, category="tool")
            session.add(parent)
            await session.commit()
            await session.refresh(parent)
            parent_id = parent.id

        report = await sync_skills(ontology)

        assert report["missing_before"] == [child_name]
        assert report["inserted"] == 1
        assert report["final_missing"] == []
        assert report["final_synced"] is True

        async with async_session_factory() as session:
            parent = await session.scalar(select(Skill).where(Skill.canonical == parent_name))
            child = await session.scalar(select(Skill).where(Skill.canonical == child_name))
            assert parent is not None
            assert child is not None
            assert parent.id == parent_id
            assert parent.category == "language"
            assert child.category == "framework"
            assert child.parent_id == parent.id
    finally:
        async with async_session_factory() as session:
            await session.execute(
                delete(Skill).where(Skill.canonical.in_([child_name, parent_name]))
            )
            await session.commit()


async def test_sync_is_idempotent_and_prevents_duplicate_canonicals() -> None:
    suffix = uuid4().hex[:8]
    name = f"pytest-sync-idempotent-{suffix}"
    ontology = SkillsOntology([SkillEntry(canonical=name, category="tool")])

    try:
        first = await sync_skills(ontology)
        second = await sync_skills(ontology)

        assert first["inserted"] == 1
        assert second["inserted"] == 0
        assert second["updated"] == 0

        async with async_session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(Skill).where(Skill.canonical == name)
            )
            assert count == 1
    finally:
        async with async_session_factory() as session:
            await session.execute(delete(Skill).where(Skill.canonical == name))
            await session.commit()


async def test_sync_reports_extra_database_skills_without_deleting_them() -> None:
    suffix = uuid4().hex[:8]
    ontology_name = f"pytest-sync-ontology-{suffix}"
    extra_name = f"pytest-sync-extra-{suffix}"
    ontology = SkillsOntology([SkillEntry(canonical=ontology_name, category="tool")])

    try:
        async with async_session_factory() as session:
            session.add(Skill(canonical=extra_name, category="platform"))
            await session.commit()

        report = await sync_skills(ontology)

        assert extra_name in report["extra_before"]
        assert extra_name in report["final_extra"]
        async with async_session_factory() as session:
            extra = await session.scalar(select(Skill).where(Skill.canonical == extra_name))
            assert extra is not None
    finally:
        async with async_session_factory() as session:
            await session.execute(
                delete(Skill).where(Skill.canonical.in_([ontology_name, extra_name]))
            )
            await session.commit()


async def test_inspection_reports_missing_and_duplicates_shape() -> None:
    suffix = uuid4().hex[:8]
    missing_name = f"pytest-sync-missing-{suffix}"
    ontology = SkillsOntology([SkillEntry(canonical=missing_name, category="tool")])

    async with async_session_factory() as session:
        report = await inspect_skill_sync(session, ontology)

    assert report["ontology_count"] == 1
    assert missing_name in report["missing_before"]
    assert report["duplicate_canonicals_before"] == []


async def test_runner_sync_validation_passes_after_production_ontology_sync() -> None:
    ontology = load_ontology()
    report = await sync_skills(ontology)

    assert report["final_missing"] == []
    assert report["final_duplicate_canonicals"] == []

    async with async_session_factory() as session:
        skill_ids = await _load_skill_ids(session, ontology)

    assert len(skill_ids) == len(ontology)
