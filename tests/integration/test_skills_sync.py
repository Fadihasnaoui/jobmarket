"""Integration tests for syncing the skills ontology into the skills table."""

from __future__ import annotations

import uuid

from sqlalchemy import delete, func, select

from jobmarket.db.models import Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.ontology import SkillEntry, SkillsOntology
from jobmarket.skills.sync import sync_skills


async def test_sync_is_idempotent_and_links_parents() -> None:
    suffix = uuid.uuid4().hex[:8]
    parent_name = f"pytest-lang-{suffix}"
    child_name = f"pytest-framework-{suffix}"

    ontology = SkillsOntology(
        [
            SkillEntry(canonical=parent_name, category="language", aliases=[], parent=None),
            SkillEntry(
                canonical=child_name, category="framework", aliases=[], parent=parent_name
            ),
        ]
    )

    try:
        result = await sync_skills(ontology)
        assert result["synced"] == 2

        async with async_session_factory() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(Skill)
                .where(Skill.canonical.in_([parent_name, child_name]))
            )
            assert count == 2

            parent_row = await session.scalar(
                select(Skill).where(Skill.canonical == parent_name)
            )
            child_row = await session.scalar(select(Skill).where(Skill.canonical == child_name))
            assert child_row.parent_id == parent_row.id

        # Re-sync must not create duplicates.
        await sync_skills(ontology)
        async with async_session_factory() as session:
            count = await session.scalar(
                select(func.count())
                .select_from(Skill)
                .where(Skill.canonical.in_([parent_name, child_name]))
            )
            assert count == 2
    finally:
        async with async_session_factory() as session:
            await session.execute(
                delete(Skill).where(Skill.canonical.in_([parent_name, child_name]))
            )
            await session.commit()


async def test_sync_updates_category_on_conflict() -> None:
    suffix = uuid.uuid4().hex[:8]
    name = f"pytest-recat-{suffix}"

    try:
        await sync_skills(SkillsOntology([SkillEntry(canonical=name, category="tool")]))
        async with async_session_factory() as session:
            row = await session.scalar(select(Skill).where(Skill.canonical == name))
            assert row.category == "tool"

        await sync_skills(SkillsOntology([SkillEntry(canonical=name, category="platform")]))
        async with async_session_factory() as session:
            row = await session.scalar(select(Skill).where(Skill.canonical == name))
            assert row.category == "platform"

            count = await session.scalar(
                select(func.count()).select_from(Skill).where(Skill.canonical == name)
            )
            assert count == 1
    finally:
        async with async_session_factory() as session:
            await session.execute(delete(Skill).where(Skill.canonical == name))
            await session.commit()
