"""Tests for dataset management infrastructure."""

from __future__ import annotations

import json
from pathlib import Path

from jobmarket.datasets.annotation import (
    empty_annotation_template,
    prepare_gold_structure,
    validate_annotation,
)
from jobmarket.datasets.duplicates import detect_duplicates
from jobmarket.datasets.evaluator import evaluate_cv_extraction
from jobmarket.datasets.inventory import write_inventory_reports
from jobmarket.datasets.quality_gates import evaluate_quality_gates
from jobmarket.datasets.registry import generate_registry
from jobmarket.datasets.splits import validate_splits


def test_dataset_inventory_reports_counts_extensions_and_empty_folders(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    reports = tmp_path / "reports"
    dataset = raw / "sample_dataset"
    empty = dataset / "empty"
    empty.mkdir(parents=True)
    (dataset / "a.txt").write_text("hello", encoding="utf-8")
    (dataset / "b.csv").write_text("x\n1\n", encoding="utf-8")

    result = write_inventory_reports(raw, reports)

    assert result[0].dataset_name == "sample_dataset"
    assert result[0].number_of_files == 2
    assert result[0].extensions == {".csv": 1, ".txt": 1}
    assert "sample_dataset/empty" in result[0].empty_folders
    assert (reports / "dataset_inventory.json").exists()
    assert (reports / "dataset_inventory.md").exists()


def test_registry_marks_unknown_license_without_invention(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    registry = tmp_path / "registry.json"
    (raw / "synthetic_resumes").mkdir(parents=True)
    (raw / "synthetic_resumes" / "resumes.csv").write_text("a\n", encoding="utf-8")

    result = generate_registry(raw, registry)

    dataset = result["datasets"][0]
    assert dataset["license"] == "UNKNOWN"
    assert dataset["license_status"] == "TO_VERIFY"
    assert dataset["contains_synthetic_profiles"] is True


def test_duplicate_detection_uses_sha_and_normalized_text(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("Email hidden Python AWS", encoding="utf-8")
    (raw / "b.txt").write_text("Email hidden Python AWS", encoding="utf-8")
    (raw / "c.txt").write_text("email HIDDEN python aws", encoding="utf-8")

    result = detect_duplicates(raw)

    assert result["sha256_duplicates"][0]["count"] == 2
    assert result["normalized_text_duplicates"][0]["count"] == 3


def test_annotation_template_and_schema_validation() -> None:
    template = empty_annotation_template()

    assert template["review"]["status"] == "MANUAL_REVIEW_REQUIRED"
    assert validate_annotation(template) == []
    broken = {"document": {}, "expected": {}, "review": {}}
    assert validate_annotation(broken)


def test_prepare_gold_structure_writes_schema_templates_and_splits(
    tmp_path: Path, monkeypatch
) -> None:
    import jobmarket.datasets.annotation as annotation
    import jobmarket.datasets.paths as paths

    monkeypatch.setattr(paths, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(annotation, "GOLD_SCHEMA_DIR", tmp_path / "gold" / "schema")
    monkeypatch.setattr(annotation, "GOLD_TEMPLATES_DIR", tmp_path / "gold" / "templates")
    monkeypatch.setattr(annotation, "GOLD_SPLITS_DIR", tmp_path / "gold" / "splits")
    monkeypatch.setattr(annotation, "ensure_dataset_dirs", lambda: None)

    prepare_gold_structure()

    assert (tmp_path / "gold" / "schema" / "cv_annotation.schema.json").exists()
    assert (tmp_path / "gold" / "templates" / "cv_annotation_template.json").exists()
    assert (tmp_path / "gold" / "splits" / "development.txt").exists()


def test_split_validation_detects_duplicates_and_missing_files(tmp_path: Path) -> None:
    splits = tmp_path / "splits"
    annotations = tmp_path / "annotations"
    documents = tmp_path / "documents"
    splits.mkdir()
    annotations.mkdir()
    documents.mkdir()
    (splits / "development.txt").write_text("cv1.txt\n", encoding="utf-8")
    (splits / "validation.txt").write_text("cv1.txt\n", encoding="utf-8")
    (splits / "test.txt").write_text("cv2.txt\n", encoding="utf-8")
    annotation = empty_annotation_template("cv1")
    annotation["document"]["relative_path"] = "cv1.txt"
    annotation["document"]["sha256"] = "abc"
    (annotations / "cv1.json").write_text(json.dumps(annotation), encoding="utf-8")
    (documents / "cv1.txt").write_text("Profile\nPython", encoding="utf-8")

    result = validate_splits(splits, annotations, documents)

    assert result["valid"] is False
    assert result["duplicate_cv_between_splits"] == ["cv1.txt"]
    assert result["missing_annotations"] == ["cv2.txt"]


def test_quality_gates_fail_when_thresholds_are_not_met() -> None:
    result = evaluate_quality_gates({"parser_success_rate": 0.5}, {"parser_success_rate": 0.9})

    assert result["passed"] is False
    assert result["failures"][0]["metric"] == "parser_success_rate"


def test_evaluator_writes_reports_without_raw_text(tmp_path: Path) -> None:
    docs = tmp_path / "documents"
    annotations = tmp_path / "annotations"
    reports = tmp_path / "reports"
    docs.mkdir()
    annotations.mkdir()
    (docs / "cv.txt").write_text("Profile\nSkills\nPython", encoding="utf-8")
    annotation = empty_annotation_template("cv")
    annotation["document"].update({"relative_path": "cv.txt", "sha256": "abc"})
    annotation["expected"].update(
        {
            "skills": ["Python"],
            "sections": ["profile", "skills"],
            "should_abstain": False,
        }
    )
    (annotations / "cv.json").write_text(json.dumps(annotation), encoding="utf-8")

    report = evaluate_cv_extraction(
        docs, annotations, reports, thresholds={"parser_success_rate": 0.1}
    )

    assert report["metrics"]["parser_success_rate"] == 1.0
    output = (reports / "cv_extraction_validation.json").read_text(encoding="utf-8")
    assert "Profile\nSkills" not in output
