"""Tests for embedding runner and semantic repository behavior."""

from __future__ import annotations

import pytest

from jobmarket.embeddings.config import EmbeddingSettings
from jobmarket.embeddings.repository import SemanticSearchFilters
from jobmarket.embeddings.runner import JobEmbeddingConfig, run_job_embedding


class FakeEncoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("boom")
        return [[1.0] + [0.0] * 383 for _ in texts]


def _settings() -> EmbeddingSettings:
    return EmbeddingSettings(
        model_name="fake-model",
        dimension=384,
        batch_size=2,
        normalize=True,
        model_revision="test",
    )


async def test_embedding_runner_dry_run_uses_pending_count_without_encoder(monkeypatch) -> None:
    async def fake_select(session, config, settings):
        return [10, 20, 30]

    monkeypatch.setattr("jobmarket.embeddings.runner._select_pending_job_ids", fake_select)

    summary = await run_job_embedding(
        JobEmbeddingConfig(limit=3, dry_run=True),
        settings=_settings(),
    )

    assert summary.selected_jobs == 3
    assert summary.embedded_jobs == 0
    assert summary.pending_jobs == 3
    assert summary.dry_run is True


async def test_embedding_runner_isolates_batch_failures(monkeypatch) -> None:
    async def fake_select(session, config, settings):
        return [10, 20]

    async def fake_load(session, batch_ids, run_id):
        return [
            {
                "job": {"id": job_id, "title": "Python", "description": "Python"},
                "enrichment": {"skills": ["Python"]},
            }
            for job_id in batch_ids
        ]

    persisted: list[int] = []

    async def fake_persist(session, job_id, vector, settings):
        persisted.append(job_id)

    monkeypatch.setattr("jobmarket.embeddings.runner._select_pending_job_ids", fake_select)
    monkeypatch.setattr("jobmarket.embeddings.runner._load_embedding_rows", fake_load)
    monkeypatch.setattr("jobmarket.embeddings.runner._persist_embedding", fake_persist)

    summary = await run_job_embedding(
        JobEmbeddingConfig(batch_size=2),
        encoder=FakeEncoder(fail=True),
        settings=_settings(),
    )

    assert summary.embedded_jobs == 0
    assert summary.pending_jobs == 2
    assert summary.failed_batches == 1
    assert persisted == []


def test_semantic_search_filters_are_explicit() -> None:
    filters = SemanticSearchFilters(
        country="fr", career_level="senior", contract_type="CDI", work_mode="remote"
    )

    assert filters.country == "fr"
    assert filters.career_level == "senior"
    assert filters.contract_type == "CDI"
    assert filters.work_mode == "remote"


def test_runner_rejects_invalid_options() -> None:
    with pytest.raises(ValueError, match="limit"):
        import asyncio

        asyncio.run(run_job_embedding(JobEmbeddingConfig(limit=0), settings=_settings()))
