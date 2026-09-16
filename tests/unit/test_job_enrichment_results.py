"""Tests for the per-job enrichment result ledger."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, UniqueConstraint

from jobmarket.db.models import JobEnrichmentResult


def _constraint_names(constraint_type: type[object]) -> set[str]:
    return {
        constraint.name or ""
        for constraint in JobEnrichmentResult.__table__.constraints
        if isinstance(constraint, constraint_type)
    }


def _load_migration() -> ModuleType:
    path = Path("alembic/versions/004_job_enrichment_results.py")
    spec = importlib.util.spec_from_file_location("migration_004_job_enrichment_results", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_success_result() -> None:
    result = JobEnrichmentResult(run_id=1, job_id=2, status="success", skill_count=3)

    assert result.status == "success"
    assert result.skill_count == 3


def test_valid_failed_result() -> None:
    result = JobEnrichmentResult(run_id=1, job_id=2, status="failed", skill_count=0)

    assert result.status == "failed"
    assert result.skill_count == 0


def test_success_with_zero_skills_is_valid() -> None:
    result = JobEnrichmentResult(run_id=1, job_id=2, status="success", skill_count=0)

    assert result.status == "success"
    assert result.skill_count == 0


def test_rejects_negative_skill_count() -> None:
    with pytest.raises(ValueError, match="skill_count must be non-negative"):
        JobEnrichmentResult(run_id=1, job_id=2, status="success", skill_count=-1)


def test_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="status must be"):
        JobEnrichmentResult(run_id=1, job_id=2, status="running", skill_count=0)


def test_unique_run_job_constraint() -> None:
    unique_names = _constraint_names(UniqueConstraint)
    assert "uq_job_enrichment_results_run_job" in unique_names

    unique = next(
        constraint
        for constraint in JobEnrichmentResult.__table__.constraints
        if constraint.name == "uq_job_enrichment_results_run_job"
    )
    assert [column.name for column in unique.columns] == ["run_id", "job_id"]


def test_check_constraints() -> None:
    check_names = _constraint_names(CheckConstraint)

    assert "ck_job_enrichment_results_status" in check_names
    assert "ck_job_enrichment_results_skill_count_nonnegative" in check_names


def test_only_run_foreign_key_cascades() -> None:
    fks = [
        constraint
        for constraint in JobEnrichmentResult.__table__.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    ]
    ondelete_by_local_column = {
        next(iter(constraint.columns)).name: constraint.ondelete for constraint in fks
    }

    assert ondelete_by_local_column == {"job_id": None, "run_id": "CASCADE"}


def test_useful_indexes_exist() -> None:
    indexes = {
        index.name: [column.name for column in index.columns]
        for index in JobEnrichmentResult.__table__.indexes
    }

    assert indexes == {
        "ix_job_enrichment_results_job_id": ["job_id"],
        "ix_job_enrichment_results_run_id": ["run_id"],
        "ix_job_enrichment_results_run_status": ["run_id", "status"],
    }
    assert all(isinstance(index, Index) for index in JobEnrichmentResult.__table__.indexes)


def test_migration_identity() -> None:
    migration = _load_migration()

    assert migration.revision == "004_job_enrichment_results"
    assert migration.down_revision == "003_enrichment_layer"
    assert callable(migration.upgrade)
    assert callable(migration.downgrade)
