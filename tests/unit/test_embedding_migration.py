"""Tests for job embedding migration metadata."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_migration() -> ModuleType:
    path = Path("alembic/versions/005_job_embeddings.py")
    spec = importlib.util.spec_from_file_location("migration_005_job_embeddings", path)
    assert spec is not None
    assert spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    return migration


def test_migration_identity_and_pgvector_operations() -> None:
    migration = _load_migration()

    assert migration.revision == "005_job_embeddings"
    assert migration.down_revision == "004_job_enrichment_results"
    assert callable(migration.upgrade)
    assert callable(migration.downgrade)
    source = Path("alembic/versions/005_job_embeddings.py").read_text()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in source
    assert "vector_cosine_ops" in source
    assert "USING hnsw" in source
