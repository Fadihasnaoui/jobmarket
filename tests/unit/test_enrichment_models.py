"""Tests for Layer 3 enrichment persistence models and migration metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from sqlalchemy import CheckConstraint, ForeignKeyConstraint, UniqueConstraint

from jobmarket.db.models import EnrichmentFailure, EnrichmentRun, JobSkill


def _constraint_names(model: type[object], constraint_type: type[object]) -> set[str]:
    return {
        constraint.name or ""
        for constraint in model.__table__.constraints
        if isinstance(constraint, constraint_type)
    }


def _load_migration() -> ModuleType:
    path = Path("alembic/versions/003_enrichment_layer.py")
    spec = importlib.util.spec_from_file_location("migration_003_enrichment_layer", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enrichment_run_constraints_and_config_jsonb() -> None:
    table = EnrichmentRun.__table__

    assert table.c.config.type.__class__.__name__ == "JSONB"
    assert table.c.matcher_version.nullable is False
    assert table.c.ontology_version.nullable is False

    check_names = _constraint_names(EnrichmentRun, CheckConstraint)
    assert "ck_enrichment_runs_run_type" in check_names
    assert "ck_enrichment_runs_status" in check_names
    assert "ck_enrichment_runs_jobs_processed_nonnegative" in check_names


def test_job_skill_run_scoped_uniqueness_and_offset_constraints() -> None:
    unique_names = _constraint_names(JobSkill, UniqueConstraint)
    check_names = _constraint_names(JobSkill, CheckConstraint)

    assert "uq_job_skills_run_job_skill_method" in unique_names
    unique = next(
        constraint
        for constraint in JobSkill.__table__.constraints
        if constraint.name == "uq_job_skills_run_job_skill_method"
    )
    assert [column.name for column in unique.columns] == [
        "run_id",
        "job_id",
        "skill_id",
        "method",
    ]
    assert "ck_job_skills_evidence_field" in check_names
    assert "ck_job_skills_start_char_nonnegative" in check_names
    assert "ck_job_skills_end_after_start" in check_names
    assert "ck_job_skills_confidence_range" in check_names


def test_enrichment_failure_run_job_stage_uniqueness() -> None:
    unique_names = _constraint_names(EnrichmentFailure, UniqueConstraint)
    check_names = _constraint_names(EnrichmentFailure, CheckConstraint)

    assert "uq_enrichment_failures_run_job_stage" in unique_names
    assert "ck_enrichment_failures_attempts_positive" in check_names


def test_only_run_foreign_keys_cascade_to_enrichment_children() -> None:
    job_skill_fks = [
        constraint
        for constraint in JobSkill.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]
    failure_fks = [
        constraint
        for constraint in EnrichmentFailure.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]

    def ondelete_by_local_column(fks: list[ForeignKeyConstraint]) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for constraint in fks:
            local_column = next(iter(constraint.columns)).name
            result[local_column] = constraint.ondelete
        return result

    assert ondelete_by_local_column(job_skill_fks) == {
        "job_id": None,
        "run_id": "CASCADE",
        "skill_id": None,
    }
    assert ondelete_by_local_column(failure_fks) == {
        "job_id": None,
        "run_id": "CASCADE",
    }


def test_enrichment_migration_identity() -> None:
    migration = _load_migration()

    assert migration.revision == "003_enrichment_layer"
    assert migration.down_revision == "002_skills_ontology"
    assert callable(migration.upgrade)
    assert callable(migration.downgrade)
