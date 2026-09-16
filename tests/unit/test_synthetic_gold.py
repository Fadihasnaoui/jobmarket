"""Tests for scenario-first synthetic gold dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import jobmarket.datasets.synthetic_gold as synthetic_gold
from jobmarket.datasets.synthetic_gold import (
    CAREERS,
    DOMAINS,
    evaluate_synthetic_cv_extraction,
    generate_synthetic_gold,
    validate_synthetic_gold,
)


def test_synthetic_gold_generation_validation_and_coverage(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    result = validate_synthetic_gold(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    counts = result.counts

    assert result.valid is True
    assert len(items) == 30
    assert counts["split_counts"] == {"development": 18, "validation": 6, "test": 6}
    assert counts["format_counts"]["pdf"] >= 16
    assert counts["format_counts"]["docx"] >= 7
    assert counts["format_counts"]["txt"] >= 7
    assert counts["language_counts"]["fr"] >= 12
    assert counts["language_counts"]["en"] >= 12
    assert counts["language_counts"]["bilingual"] >= 4
    assert DOMAINS.issubset(counts["domain_counts"])
    assert CAREERS.issubset(counts["career_level_counts"])
    assert counts["quality_counts"] == {"abstain": 4, "clear": 22, "difficult": 4}
    assert counts["positive_ambiguous_alias_cases"] >= 4
    assert counts["negative_ambiguous_alias_cases"] >= 8


def test_synthetic_annotations_are_scenario_first_and_consistent(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    hashes = [item["sha256"] for item in manifest["items"]]

    assert len(hashes) == len(set(hashes))
    for item in manifest["items"]:
        scenario = json.loads((root / item["scenario"]).read_text(encoding="utf-8"))
        annotation = json.loads((root / item["annotation"]).read_text(encoding="utf-8"))
        expected = annotation["expected"]
        assert annotation["synthetic_metadata"]["annotation_origin"] == "SYNTHETIC_SCENARIO"
        assert set(expected["skills"]).issubset(set(scenario["explicit_skills"]))
        assert not (set(expected["skills"]) & set(expected["forbidden_skills"]))
        assert annotation["synthetic_metadata"]["contains_real_pii"] is False


def test_synthetic_test_split_is_protected(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)

    with pytest.raises(ValueError, match="test evaluation requires"):
        evaluate_synthetic_cv_extraction(root=root, split="test")


def test_synthetic_evaluation_writes_safe_reports_without_test_details(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    report = evaluate_synthetic_cv_extraction(root=root, split="development")

    assert report["split"] == "development"
    output = (root / "reports" / "development_evaluation.json").read_text(encoding="utf-8")
    assert "candidate001@example.test" not in output
    assert "Python" not in output
    assert "Example University" not in output


def test_synthetic_generation_does_not_touch_real_gold_raw_or_run_files(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    real_gold = tmp_path / "gold" / "documents"
    raw = tmp_path / "raw"
    run_marker = tmp_path / "run_342.marker"
    real_gold.mkdir(parents=True)
    raw.mkdir()
    run_marker.write_text("unchanged", encoding="utf-8")

    generate_synthetic_gold(root)

    assert list(real_gold.iterdir()) == []
    assert list(raw.iterdir()) == []
    assert run_marker.read_text(encoding="utf-8") == "unchanged"

def test_synthetic_evaluator_uses_education_extraction_not_section_presence() -> None:
    expected = {
        "sections": ["education"],
        "skills": [],
        "expected_job_families": [],
        "domains": [],
        "forbidden_skills": [],
        "education": ["Bachelor in Finance"],
        "experience_count": 0,
        "internship_count": 0,
        "career_level": "unknown",
        "status": "success",
        "should_abstain": False,
    }

    row = synthetic_gold._row_scores(
        {},
        expected,
        {"education"},
        set(),
        set(),
        set(),
        False,
        False,
        0,
        0,
        "unknown",
    )

    assert row["education_match"] is False


def test_synthetic_evaluator_resolves_expected_vocabularies() -> None:
    expected = {
        "sections": [],
        "skills": ["NLP", "APIs RESTful"],
        "expected_job_families": [],
        "domains": ["Artificial Intelligence"],
        "forbidden_skills": [],
        "education": [],
        "experience_count": 0,
        "internship_count": 0,
        "career_level": "mid level",
        "status": "success",
        "should_abstain": False,
    }

    row = synthetic_gold._row_scores(
        {},
        expected,
        set(),
        {"Natural Language Processing", "API Design"},
        set(),
        {"AI"},
        False,
        False,
        0,
        0,
        "mid",
    )

    assert row["skill"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert row["domain"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert row["career_level_match"] is True

def test_synthetic_evaluator_does_not_count_parse_errors_as_success() -> None:
    expected = {
        "sections": [],
        "skills": [],
        "expected_job_families": [],
        "domains": [],
        "forbidden_skills": [],
        "education": [],
        "experience_count": 0,
        "internship_count": 0,
        "career_level": "unknown",
        "status": "success",
        "should_abstain": False,
    }

    row = synthetic_gold._row_scores(
        {"parser_success": False, "error": "EmptyDocumentError"},
        expected,
        set(),
        set(),
        set(),
        set(),
        False,
        False,
        0,
        0,
        "unknown",
    )

    assert row["expected_status_match"] is False

def test_synthetic_canonical_status_policy() -> None:
    assert synthetic_gold.canonical_expected_status("abstain") == "insufficient_profile_evidence"
    assert synthetic_gold.canonical_expected_status("ocr_required") == "ocr_required"
    assert (
        synthetic_gold.canonical_runtime_status(parser_success=True, layout="single column")
        == "success"
    )
    assert synthetic_gold.canonical_runtime_status(
        parser_success=True,
        layout="single column",
        abstention_reason="insufficient_profile_evidence",
    ) == "insufficient_profile_evidence"
    assert synthetic_gold.canonical_runtime_status(
        parser_success=False,
        layout="scanned-image-like PDF requiring OCR",
        error_type="EmptyDocumentError",
    ) == "ocr_required"
    assert synthetic_gold.canonical_runtime_status(
        parser_success=False,
        layout="native-text single column",
        error_type="MalformedDocumentError",
    ) == "parse_error"


def test_synthetic_status_confusion_matrix_and_ocr_policy(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    report = synthetic_gold.write_synthetic_consistency_reports(root)

    assert (root / "reports" / "status_diagnosis.json").exists()
    counts = report["status"]["aggregate_counts"]
    assert counts["expected_ocr_required__actual_ocr_required"] == 1
    assert all(row["cv_id"] != "cv_synth_025" for row in report["status"]["rows"])


def test_synthetic_generated_experience_and_education_patterns(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    item = next(row for row in manifest["items"] if row["cv_id"] == "cv_synth_001")
    scenario = json.loads((root / item["scenario"]).read_text(encoding="utf-8"))
    text = synthetic_gold._document_text(scenario)

    assert "Commercial Support - Example Company 001 | Jan 2021 - Dec 2022 |" in text
    assert "Bachelor in data science - Example University 001" in text
    assert "Status: Graduated" in text


def test_synthetic_abstention_scenarios_remain_insufficient(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    abstain_items = [row for row in manifest["items"] if row["expected_status"] == "abstain"]

    assert abstain_items
    for item in abstain_items:
        scenario = json.loads((root / item["scenario"]).read_text(encoding="utf-8"))
        text = synthetic_gold._document_text(scenario)
        assert "Jan 2021 - Dec 2022" not in text
        assert "Status: Graduated" not in text


def test_synthetic_repair_updates_only_permitted_split_hashes(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    manifest_before = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    test_hashes_before = {
        item["cv_id"]: item["sha256"]
        for item in manifest_before["items"]
        if item["split"] == "test"
    }

    result = synthetic_gold.repair_synthetic_gold_development_validation(root)
    manifest_after = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    test_hashes_after = {
        item["cv_id"]: item["sha256"]
        for item in manifest_after["items"]
        if item["split"] == "test"
    }

    assert result["test_hashes_unchanged"] is True
    assert test_hashes_after == test_hashes_before
    assert not set(result["changed_documents"]) & set(test_hashes_after)

def test_synthetic_skill_and_domain_audits_are_metadata_only(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    synthetic_gold.repair_synthetic_gold_development_validation(root)

    report = synthetic_gold.write_synthetic_skill_domain_audits(root)

    assert (root / "reports" / "skill_error_audit.json").exists()
    assert (root / "reports" / "domain_error_audit.json").exists()
    assert report["skill"]["errors"]
    assert report["domain"]["errors"]
    output = (root / "reports" / "skill_error_audit.json").read_text(encoding="utf-8")
    assert "candidate001@example.test" not in output
    assert "Worked on" not in output
    assert all(row["cv_id"] != "cv_synth_025" for row in report["skill"]["rows"])
    assert all(row["cv_id"] != "cv_synth_025" for row in report["domain"]["rows"])


def test_synthetic_skill_audit_documents_safe_normalization_policy(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    synthetic_gold.repair_synthetic_gold_development_validation(root)

    report = synthetic_gold.write_synthetic_skill_domain_audits(root)
    policy = report["skill"]["policy"]
    rows = {row["cv_id"]: row for row in report["skill"]["rows"]}

    assert "canonical_equivalent" in policy
    assert "Natural Language Processing" in rows["cv_synth_021"]["true_positives"]
    assert "API Design" in rows["cv_synth_022"]["true_positives"]
    assert policy["parent_child_equivalent"].startswith("Not accepted")


def test_synthetic_skill_audit_metrics_and_classifications(tmp_path: Path) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    synthetic_gold.repair_synthetic_gold_development_validation(root)

    report = synthetic_gold.write_synthetic_skill_domain_audits(root)
    metrics = report["skill"]["metrics"]
    classifications = {
        row["classification"] for row in report["skill"]["error_classification_summary"]
    }

    assert metrics["micro"]["precision"] >= 0.0
    assert metrics["macro"]["f1"] >= 0.0
    assert metrics["inferred_skill_metrics"] == "NOT_IMPLEMENTED"
    assert "annotation_too_strict" in classifications


def test_synthetic_domain_audit_no_longer_reports_commercial_support_pollution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "gold_synthetic"
    generate_synthetic_gold(root)
    synthetic_gold.repair_synthetic_gold_development_validation(root)

    report = synthetic_gold.write_synthetic_skill_domain_audits(root)
    commercial_support_errors = [
        row
        for row in report["domain"]["errors"]
        if row["domain"] == "Commercial Support" and row["direction"] == "false_positive"
    ]
    false_positive_labels = {
        row["label"] for row in report["domain"]["top_false_positive_domains"]
    }

    assert commercial_support_errors == []
    assert "Commercial Support" not in false_positive_labels
