"""Tests for CV CLI and privacy-safe formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from jobmarket.cli import app
from jobmarket.cv.profile import (
    CvMatchResult,
    CvProfile,
    CvSkill,
    DocumentPage,
    EvidenceSpan,
    ExtractionVersions,
    GuardedExtractionReport,
    JobRecommendation,
    ParsedDocument,
)

runner = CliRunner()


def test_cv_match_cli_invokes_service_with_defaults(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("SECRET CV TEXT Python", encoding="utf-8")
    calls: dict[str, Any] = {}

    async def fake_match_cv_document(
        path_arg: str, *, top: int, run_id: int, mode: str = "lexical", alpha: float = 0.6
    ) -> CvMatchResult:
        calls["path"] = path_arg
        calls["top"] = top
        calls["run_id"] = run_id
        calls["mode"] = mode
        calls["alpha"] = alpha
        return _result(path)

    monkeypatch.setattr("jobmarket.cli.match_cv_document", fake_match_cv_document)

    result = runner.invoke(app, ["cv", "match", str(path), "--top", "3"])

    assert result.exit_code == 0
    assert calls == {"path": str(path), "top": 3, "run_id": 113, "mode": "lexical", "alpha": 0.6}
    assert "Enrichment run 113 contains only 1,000 validation jobs" in result.output
    assert "Python" in result.output
    assert "SECRET CV TEXT" not in result.output


def test_cv_match_cli_json_is_privacy_safe(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("SECRET CV TEXT Python", encoding="utf-8")

    async def fake_match_cv_document(
        path_arg: str, *, top: int, run_id: int, mode: str = "lexical", alpha: float = 0.6
    ) -> CvMatchResult:
        return _result(Path(path_arg))

    monkeypatch.setattr("jobmarket.cli.match_cv_document", fake_match_cv_document)

    result = runner.invoke(app, ["cv", "match", str(path), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["enrichment_run_id"] == 113
    assert "text" not in payload["profile"]["document"]
    assert "SECRET CV TEXT" not in result.output


def test_cv_match_cli_rejects_guarded_mode_until_offline_input_exists(tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("Python", encoding="utf-8")

    result = runner.invoke(app, ["cv", "match", str(path), "--allow-guarded-candidates"])

    assert result.exit_code == 1
    assert "guarded candidate input is not exposed" in result.output


def test_embed_jobs_cli_invokes_runner(monkeypatch: Any) -> None:
    calls: dict[str, Any] = {}

    async def fake_run_job_embedding(config: Any) -> Any:
        calls["config"] = config

        class Summary:
            model_name = "fake-model"
            embedding_version = "job-embedding-v1:fake"
            dimension = 384
            selected_jobs = 5
            embedded_jobs = 0
            pending_jobs = 5
            failed_batches = 0
            dry_run = True
            elapsed_seconds = 0.1
            average_jobs_per_second = 0.0

        return Summary()

    monkeypatch.setattr("jobmarket.cli.run_job_embedding", fake_run_job_embedding)

    result = runner.invoke(app, ["embed", "jobs", "--limit", "5", "--dry-run"])

    assert result.exit_code == 0
    assert calls["config"].limit == 5
    assert calls["config"].dry_run is True
    assert "selected_jobs: 5" in result.output


def _result(path: Path) -> CvMatchResult:
    document = ParsedDocument(
        filename=path.name,
        mime_type="text/plain",
        text="SECRET CV TEXT Python",
        page_count=None,
        pages=[
            DocumentPage(page=None, text="SECRET CV TEXT Python", start_offset=0, end_offset=21)
        ],
        warnings=[],
        file_hash="a" * 64,
        file_size=21,
    )
    versions = ExtractionVersions(
        parser_version=document.parser_version,
        matcher_version="deterministic-alias-matcher-v2",
        ontology_version="ontology-test",
        guard_version="cv-llm-guard-v1",
    )
    skill = CvSkill(
        canonical_skill="Python",
        ontology_skill_id=1,
        matched_alias="Python",
        extraction_method="deterministic",
        evidence=EvidenceSpan(
            evidence_text="Python",
            page=None,
            document_start=15,
            document_end=21,
            page_start=None,
            page_end=None,
        ),
        matcher_version=versions.matcher_version,
        ontology_version=versions.ontology_version,
        validation_status="deterministic",
        confidence=1.0,
    )
    profile = CvProfile(document=document, versions=versions, skills=[skill])
    guard = GuardedExtractionReport(
        parser_version=document.parser_version,
        matcher_version=versions.matcher_version,
        ontology_version=versions.ontology_version,
        guard_version="cv-llm-guard-v1",
        deterministic_skills=[skill],
        submitted_llm_candidates=0,
        accepted_llm_candidates=[],
        rejected_candidates=[],
        evidence_integrity_violations=[],
        document_warnings=[],
        final_skills=[skill],
    )
    recommendation = JobRecommendation(
        job_id=1,
        title="Python Developer",
        company="ExampleCo",
        location="Remote",
        final_score=1.0,
        coverage_score=1.0,
        cv_overlap_score=1.0,
        cv_skill_count=1,
        job_skill_count=1,
        matched_skill_count=1,
        matched_skills=["Python"],
        missing_skills=[],
        enrichment_run_id=113,
        scoring_version="cv-job-scorer-v1",
        explanation="Matched 1 of 1 job skills; coverage=1.000, cv_overlap=1.000.",
    )
    return CvMatchResult(
        profile=profile,
        guard_report=guard,
        recommendations=[recommendation],
        enrichment_run_id=113,
        validation_run_warning=(
            "Enrichment run 113 contains only 1,000 validation jobs; recommendations are MVP "
            "results over that subset, not the full corpus."
        ),
    )


def test_cv_match_human_output_includes_abstention_status(tmp_path: Path) -> None:
    result = _result(tmp_path / "cv.txt").model_copy(
        update={
            "status": "insufficient_profile_evidence",
            "message": "CV extraction confidence is too low for reliable recommendations.",
            "recommendations": [],
        }
    )

    from jobmarket.cv.reporting import format_cv_match_result

    output = format_cv_match_result(result)

    assert "status:                insufficient_profile_evidence" in output
    assert "message:               CV extraction confidence is too low" in output
    assert "Ranked jobs" in output
    assert "SECRET CV TEXT" not in output


def test_cv_match_json_includes_status_without_raw_text(tmp_path: Path) -> None:
    result = _result(tmp_path / "cv.txt").model_copy(update={"status": "success"})

    from jobmarket.cv.reporting import cv_match_result_to_json

    payload = json.loads(cv_match_result_to_json(result))

    assert payload["status"] == "success"
    assert "text" not in payload["profile"]["document"]
    assert "SECRET CV TEXT" not in json.dumps(payload)


def test_recommend_from_cv_cli_invokes_workflow(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("SECRET CV TEXT Python", encoding="utf-8")
    calls: dict[str, Any] = {}

    async def fake_workflow(request: Any) -> Any:
        calls["request"] = request
        from jobmarket.cv.workflow import CvRecommendationResponse, ExtractedProfileOutput

        return CvRecommendationResponse(
            document_status="parsed",
            canonical_extraction_status="success",
            ocr_required=False,
            profile=ExtractedProfileOutput(
                filename=path.name,
                mime_type="text/plain",
                document_status="parsed",
                canonical_extraction_status="success",
                ocr_required=False,
                canonical_skills=["Python"],
            ),
            recommendations=[],
        )

    monkeypatch.setattr("jobmarket.cli.run_cv_recommendation_workflow", fake_workflow)

    result = runner.invoke(
        app,
        [
            "recommend-from-cv",
            str(path),
            "--limit",
            "3",
            "--country",
            "FR",
            "--contract-type",
            "cdi",
        ],
    )

    assert result.exit_code == 0
    assert calls["request"].limit == 3
    assert calls["request"].filters.country == "FR"
    assert calls["request"].filters.contract_type == "cdi"
    assert "SECRET CV TEXT" not in result.output


def test_cv_recommend_cli_json_with_confirmed_profile(monkeypatch: Any, tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("Python", encoding="utf-8")
    confirmed = tmp_path / "confirmed.json"
    confirmed.write_text(
        json.dumps({"canonical_skills": ["Python"], "career_level": "junior"}),
        encoding="utf-8",
    )

    async def fake_workflow(request: Any) -> Any:
        from jobmarket.cv.workflow import CvRecommendationResponse, ExtractedProfileOutput

        assert request.confirmed_profile is not None
        assert request.confirmed_profile.canonical_skills == ["Python"]
        return CvRecommendationResponse(
            document_status="parsed",
            canonical_extraction_status="success",
            ocr_required=False,
            profile=ExtractedProfileOutput(
                filename=path.name,
                mime_type="text/plain",
                document_status="parsed",
                canonical_extraction_status="success",
                ocr_required=False,
                canonical_skills=["Python"],
            ),
            recommendations=[],
        )

    monkeypatch.setattr("jobmarket.cli.run_cv_recommendation_workflow", fake_workflow)

    result = runner.invoke(
        app,
        ["cv", "recommend", str(path), "--confirmed-profile", str(confirmed), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["canonical_extraction_status"] == "success"
    assert "SECRET CV TEXT" not in result.output


def test_recommend_from_cv_cli_rejects_invalid_limit(tmp_path: Path) -> None:
    path = tmp_path / "cv.txt"
    path.write_text("Python", encoding="utf-8")

    result = runner.invoke(app, ["recommend-from-cv", str(path), "--limit", "0"])

    assert result.exit_code != 0
    assert "--limit must be positive" in result.output
