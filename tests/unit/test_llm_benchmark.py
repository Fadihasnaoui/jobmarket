# ruff: noqa: E501
"""Tests for controlled Groq benchmark infrastructure; no network access."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import structlog
from typer.testing import CliRunner

from jobmarket.cli import app
from jobmarket.llm_benchmark import (
    BenchmarkJob,
    BenchmarkRecord,
    BenchmarkValidation,
    GroqCallResult,
    call_groq_job_extraction,
    compute_metrics,
    run_groq_benchmark,
    select_sample_from_jobs,
    validate_groq_response,
)

runner = CliRunner()


def _job(
    *,
    job_id: int = 1,
    title: str = "Data Engineer",
    description: str = "Build pipelines with Python. Remote role.",
    contract_type: str | None = None,
    is_remote: bool | None = None,
    skill_count: int = 0,
    deterministic_skills: tuple[str, ...] = (),
    country: str | None = "fr",
    source: str = "adzuna",
) -> BenchmarkJob:
    return BenchmarkJob(
        job_id=job_id,
        title=title,
        description=description,
        country=country,
        city="Paris",
        contract_type=contract_type,
        is_remote=is_remote,
        skill_count=skill_count,
        deterministic_skills=deterministic_skills,
        source=source,
    )


def _raw(**overrides: object) -> str:
    payload: dict[str, object] = {
        "skills": [
            {
                "name": "Python",
                "evidence": "Python",
                "confidence": 0.9,
                "explicit": True,
            }
        ],
        "career_level": {"value": "unknown", "evidence": None, "confidence": 1.0},
        "experience_requirement": {
            "minimum_years": None,
            "maximum_years": None,
            "evidence": None,
            "confidence": 1.0,
        },
        "work_mode": {"value": "unknown", "evidence": None, "confidence": 1.0},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_structured_response_accepts_ontology_skill() -> None:
    result = validate_groq_response(_job(), GroqCallResult(_raw()))

    assert result.valid_json is True
    assert [skill.canonical for skill in result.accepted_skills] == ["Python"]
    assert result.rejected_items == []


def test_malformed_json_is_rejected() -> None:
    result = validate_groq_response(_job(), GroqCallResult("not json"))

    assert result.valid_json is False
    assert result.rejected_items[0].reason.startswith("invalid_json_schema")


def test_schema_validation_canonicalizes_common_llm_enum_variants() -> None:
    raw = _raw(
        skills=[{"name": "Python", "evidence": "Python", "confidence": 0.9}],
        career_level={"value": "Senior", "evidence": "Senior", "confidence": 0.9, "explicit": True},
        experience_requirement={
            "minimum_years": "Unknown",
            "maximum_years": "Unknown",
            "evidence": None,
            "confidence": None,
        },
        work_mode={"value": "On-site", "evidence": "On-site", "confidence": 0.9, "explicit": True},
    )
    job = _job(title="Python Engineer", description="Python. Senior role. On-site.")

    result = validate_groq_response(job, GroqCallResult(raw))

    assert result.valid_json is True
    assert result.accepted_career_level == "senior"
    assert result.accepted_work_mode == "on_site"



def test_hallucinated_evidence_is_rejected() -> None:
    result = validate_groq_response(_job(), GroqCallResult(_raw(skills=[{
        "name": "Python", "evidence": "not in job", "confidence": 0.9, "explicit": True
    }])))

    assert result.rejected_items[0].reason == "hallucinated_evidence"


def test_unknown_ontology_skill_is_reported_unmapped() -> None:
    result = validate_groq_response(_job(), GroqCallResult(_raw(skills=[{
        "name": "ImaginaryDB", "evidence": "Python", "confidence": 0.9, "explicit": True
    }])))

    assert result.unmapped_skills[0].reason == "unknown_ontology_skill"


def test_duplicate_and_deterministic_skills_are_rejected() -> None:
    duplicate = validate_groq_response(_job(description="Python Python"), GroqCallResult(_raw(skills=[
        {"name": "Python", "evidence": "Python", "confidence": 0.9, "explicit": True},
        {"name": "python3", "evidence": "Python", "confidence": 0.9, "explicit": True},
    ])))
    deterministic = validate_groq_response(_job(deterministic_skills=("Python",)), GroqCallResult(_raw()))

    assert duplicate.rejected_items[0].reason == "duplicate_skill"
    assert deterministic.rejected_items[0].reason == "already_deterministic"


def test_deterministic_conflicts_and_metadata_conflicts_are_rejected() -> None:
    raw = _raw(
        career_level={"value": "junior", "evidence": "Senior", "confidence": 0.9},
        work_mode={"value": "remote", "evidence": "Remote", "confidence": 0.9},
    )
    job = _job(title="Senior Data Engineer", description="Senior role. Remote role.", is_remote=False)

    result = validate_groq_response(job, GroqCallResult(raw))

    reasons = {item.field: item.reason for item in result.rejected_items}
    assert reasons["career_level"] == "deterministic_conflict"
    assert reasons["work_mode"] == "metadata_conflict"


def test_timeout_and_rate_limit_failures_are_recorded() -> None:
    timeout = validate_groq_response(_job(), GroqCallResult("", error="timeout: boom"))
    rate_limit = validate_groq_response(_job(), GroqCallResult("", error="rate_limit: slow down"))

    assert timeout.valid_json is False
    assert timeout.rejected_items[0].reason.startswith("timeout")
    assert rate_limit.rejected_items[0].reason.startswith("rate_limit")


def test_call_groq_job_extraction_uses_mock_client_and_tokens() -> None:
    client = MagicMock()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=_raw()))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    client.chat.completions.create.return_value = response

    result = call_groq_job_extraction(_job(), client=client, model="test-model")

    assert result.raw_response
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert client.chat.completions.create.call_count == 1


def test_deterministic_sample_selection_is_stable_and_balanced() -> None:
    jobs = [
        _job(job_id=1, skill_count=0, description="short"),
        _job(job_id=2, skill_count=1, description="short Python"),
        _job(job_id=3, skill_count=6, description="x" * 2000),
        _job(job_id=4, title="Senior Engineer", skill_count=3),
        _job(job_id=5, title="Stage Data", skill_count=2),
    ]

    first = select_sample_from_jobs(jobs, 4)
    second = select_sample_from_jobs(list(reversed(jobs)), 4)

    assert [job.job_id for job in first] == [job.job_id for job in second]
    assert len(first) == 4


def test_phase_b_sample_selection_has_required_strata_and_nl_it_diagnostic() -> None:
    jobs = [
        _job(job_id=index, source="adzuna", country="nl" if index <= 10 else "it", description="x" * 500)
        for index in range(1, 21)
    ]
    jobs.extend(
        _job(job_id=index, source="adzuna", country="fr", description="x" * 500)
        for index in range(21, 101)
    )
    jobs.extend(
        _job(job_id=index, source="remoteok", country=None, title="Software Engineer", description="x" * 2000)
        for index in range(101, 141)
    )

    sample = select_sample_from_jobs(list(reversed(jobs)), 100)

    assert len(sample) == 100
    assert sum(1 for job in sample if job.stratum == "truncated_adzuna") == 70
    assert sum(1 for job in sample if job.stratum == "full_text") == 30
    assert sum(1 for job in sample if job.diagnostic_group == "nl_it_diagnostic") == 15
    assert [job.job_id for job in sample] == sorted(job.job_id for job in sample)


def test_metrics_separate_zero_skill_and_nonzero_accepted_additions() -> None:
    zero = _job(job_id=1, skill_count=0)
    nonzero = _job(job_id=2, skill_count=2, deterministic_skills=("SQL",))
    records = [
        BenchmarkRecord(
            zero,
            GroqCallResult(_raw(), prompt_tokens=10, completion_tokens=5, latency_ms=10),
            BenchmarkValidation(
                True,
                accepted_skills=[
                    validate_groq_response(zero, GroqCallResult(_raw())).accepted_skills[0]
                ],
            ),
        ),
        BenchmarkRecord(
            nonzero,
            GroqCallResult(_raw(), prompt_tokens=10, completion_tokens=5, latency_ms=20),
            BenchmarkValidation(
                True,
                accepted_skills=[
                    validate_groq_response(nonzero, GroqCallResult(_raw())).accepted_skills[0]
                ],
            ),
        ),
    ]

    metrics = compute_metrics(records)

    assert metrics.accepted_additions_by_existing_skill_status == {
        "zero_skill_jobs": 1,
        "nonzero_skill_jobs": 1,
    }
    assert metrics.jobs_with_accepted_additions_by_existing_skill_status == {
        "zero_skill_jobs": 1,
        "nonzero_skill_jobs": 1,
    }


@pytest.mark.asyncio
async def test_resume_reuses_cache_and_does_not_call_groq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = {"count": 0}
    async def fake_load_run_jobs(*args: object, **kwargs: object) -> list[BenchmarkJob]:
        return [_job()]

    monkeypatch.setattr("jobmarket.llm_benchmark.load_run_jobs", fake_load_run_jobs)

    def fake_call(*args: object, **kwargs: object) -> GroqCallResult:
        calls["count"] += 1
        return GroqCallResult(_raw(), prompt_tokens=10, completion_tokens=5, latency_ms=1)

    monkeypatch.setattr("jobmarket.llm_benchmark.call_groq_job_extraction", fake_call)
    await run_groq_benchmark(sample_size=1, output_dir=tmp_path, client=object())
    await run_groq_benchmark(sample_size=1, output_dir=tmp_path, resume=True, client=object())

    assert calls["count"] == 1
    assert (tmp_path / "metrics.json").exists()


@pytest.mark.asyncio
async def test_dry_run_writes_reports_without_calling_groq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_load_run_jobs(*args: object, **kwargs: object) -> list[BenchmarkJob]:
        return [_job()]

    monkeypatch.setattr("jobmarket.llm_benchmark.load_run_jobs", fake_load_run_jobs)
    call = MagicMock()
    monkeypatch.setattr("jobmarket.llm_benchmark.call_groq_job_extraction", call)

    result = await run_groq_benchmark(sample_size=1, output_dir=tmp_path, dry_run=True)

    assert call.call_count == 0
    assert result.metrics.total_groq_calls == 0
    assert (tmp_path / "manual_review_template.csv").exists()


def test_cli_dry_run_invokes_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_run_groq_benchmark(**kwargs: object) -> object:
        class Result:
            records = [_job()]
            output_dir = tmp_path
            metrics = SimpleNamespace(
                total_groq_calls=0,
                valid_json_rate=1.0,
                guard_acceptance_rate=0.0,
                guard_rejection_rate=0.0,
                mean_accepted_skills_added_per_job=0.0,
                estimated_total_cost_usd=0.0,
            )
        return Result()

    monkeypatch.setattr("jobmarket.cli.run_groq_benchmark", fake_run_groq_benchmark)

    result = runner.invoke(app, ["llm", "benchmark", "--dry-run", "--output-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "total_groq_calls: 0" in result.output
    _reset_structlog_stdout()


def _reset_structlog_stdout() -> None:
    logging.basicConfig(format="%(message)s", stream=sys.__stdout__, level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.__stdout__),
        cache_logger_on_first_use=False,
    )
