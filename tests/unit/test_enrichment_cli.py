"""Tests for enrichment CLI commands."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from typer.testing import CliRunner

from jobmarket.cli import app
from jobmarket.reporting.enrichment import EnrichmentRunReport
from jobmarket.skills.enrichment import (
    EnrichmentConfigError,
    MatcherEnrichmentSummary,
    OntologySyncError,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def reset_structlog_after_cli() -> None:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


def _summary(status: str = "success") -> MatcherEnrichmentSummary:
    return MatcherEnrichmentSummary(
        run_id=123,
        status=status,
        jobs_total=10,
        jobs_processed=10,
        jobs_succeeded=9 if status == "partial" else 10,
        jobs_failed=1 if status == "partial" else 0,
        jobs_with_skills=6,
        zero_skill_jobs=3 if status == "partial" else 4,
    )


def _report(status: str = "success") -> EnrichmentRunReport:
    now = datetime.now(UTC)
    return EnrichmentRunReport(
        run_id=123,
        run_type="matcher",
        status=status,
        matcher_version="matcher-test",
        ontology_version="ontology-test",
        config={"limit": 10, "batch_size": 5},
        started_at=now,
        updated_at=now,
        finished_at=now,
        error=None,
        jobs_total=10,
        jobs_processed=10,
        jobs_succeeded=9 if status == "partial" else 10,
        jobs_failed=1 if status == "partial" else 0,
        jobs_with_skills=6,
        zero_skill_jobs=3 if status == "partial" else 4,
        persisted_skill_rows=12,
        mean_skills_per_successful_job=1.2,
        median_skills_per_successful_job=1.0,
        mean_skills_among_jobs_with_skills=2.0,
        zero_skill_rate=0.4,
        failure_rate=0.0,
    )


def test_run_matcher_cli_passes_all_options(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    async def fake_run(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        calls["config"] = config.to_run_config()
        calls["resume_run_id"] = resume_run_id
        calls["force"] = force
        return _summary()

    async def fake_report(run_id: int, *, top_n: int = 20) -> EnrichmentRunReport:
        calls["report_run_id"] = run_id
        calls["top_n"] = top_n
        return _report()

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_run)
    monkeypatch.setattr("jobmarket.cli.collect_enrichment_report", fake_report)

    result = runner.invoke(
        app,
        [
            "enrichment",
            "run-matcher",
            "--source",
            "adzuna",
            "--country",
            "FR",
            "--min-job-id",
            "10",
            "--max-job-id",
            "99",
            "--limit",
            "10",
            "--batch-size",
            "5",
            "--resume-run-id",
            "123",
        ],
    )

    assert result.exit_code == 0
    assert calls["config"] == {
        "source": "adzuna",
        "country": "FR",
        "min_job_id": 10,
        "max_job_id": 99,
        "limit": 10,
        "batch_size": 5,
    }
    assert calls["resume_run_id"] == 123
    assert calls["force"] is False
    assert "run_id: 123" in result.output
    assert "persisted_skill_rows: 12" in result.output


def test_run_matcher_cli_force_and_yes(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    async def fake_run(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        calls["limit"] = config.limit
        calls["force"] = force
        return _summary()

    async def fake_report(_run_id: int, *, top_n: int = 20) -> EnrichmentRunReport:
        return _report()

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_run)
    monkeypatch.setattr("jobmarket.cli.collect_enrichment_report", fake_report)

    result = runner.invoke(app, ["enrichment", "run-matcher", "--force", "--yes"])

    assert result.exit_code == 0
    assert calls == {"limit": None, "force": True}


def test_run_matcher_cli_rejects_resume_force_conflict() -> None:
    result = runner.invoke(
        app,
        ["enrichment", "run-matcher", "--resume-run-id", "1", "--force", "--limit", "10"],
    )

    assert result.exit_code != 0
    assert "cannot be used with --force" in result.output


def test_run_matcher_cli_requires_confirmation_for_unlimited(monkeypatch: Any) -> None:
    called = False

    async def fake_run(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        nonlocal called
        called = True
        return _summary()

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_run)

    result = runner.invoke(app, ["enrichment", "run-matcher"], input="n\n")

    assert result.exit_code != 0
    assert called is False


def test_run_matcher_cli_requires_confirmation_for_large_limit(monkeypatch: Any) -> None:
    called = False

    async def fake_run(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        nonlocal called
        called = True
        return _summary()

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_run)

    result = runner.invoke(app, ["enrichment", "run-matcher", "--limit", "1001"], input="n\n")

    assert result.exit_code != 0
    assert called is False


def test_run_matcher_cli_prints_partial_and_errors(monkeypatch: Any) -> None:
    async def fake_partial(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        return _summary("partial")

    async def fake_report(_run_id: int, *, top_n: int = 20) -> EnrichmentRunReport:
        return _report("partial")

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_partial)
    monkeypatch.setattr("jobmarket.cli.collect_enrichment_report", fake_report)
    result = runner.invoke(app, ["enrichment", "run-matcher", "--limit", "10"])
    assert result.exit_code == 0
    assert "status: partial" in result.output
    assert "jobs_failed: 1" in result.output

    async def fake_error(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        raise OntologySyncError("skills out of sync")

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_error)
    result = runner.invoke(app, ["enrichment", "run-matcher", "--limit", "10"])
    assert result.exit_code == 1
    assert "skills out of sync" in result.output

    async def fake_config_error(config: Any, *, resume_run_id: int | None, force: bool) -> Any:
        raise EnrichmentConfigError("bad resume")

    monkeypatch.setattr("jobmarket.cli.run_matcher_enrichment", fake_config_error)
    result = runner.invoke(app, ["enrichment", "run-matcher", "--limit", "10"])
    assert result.exit_code == 1
    assert "bad resume" in result.output


def test_report_cli_human_and_json(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    async def fake_report(run_id: int, *, top_n: int = 20) -> EnrichmentRunReport:
        calls["run_id"] = run_id
        calls["top_n"] = top_n
        return _report()

    monkeypatch.setattr("jobmarket.cli.collect_enrichment_report", fake_report)

    human = runner.invoke(app, ["enrichment", "report", "123", "--top-n", "7"])
    assert human.exit_code == 0
    assert calls == {"run_id": 123, "top_n": 7}
    assert "Enrichment run report" in human.output

    json_result = runner.invoke(app, ["enrichment", "report", "123", "--json"])
    assert json_result.exit_code == 0
    assert '"run_id"' in json_result.output


def test_report_cli_error(monkeypatch: Any) -> None:
    async def fake_report(run_id: int, *, top_n: int = 20) -> EnrichmentRunReport:
        raise ValueError("enrichment run 999 does not exist")

    monkeypatch.setattr("jobmarket.cli.collect_enrichment_report", fake_report)
    result = runner.invoke(app, ["enrichment", "report", "999"])

    assert result.exit_code == 1
    assert "does not exist" in result.output
