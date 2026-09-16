"""Tests for generated run-75 matcher quality audit exports."""

from __future__ import annotations

import csv
from pathlib import Path

from jobmarket.reporting.enrichment_quality import (
    AMBIGUOUS_ALIAS_REVIEW_FIELDS,
    EVIDENCE_INTEGRITY_FIELDS,
    HIGH_SKILL_COUNT_REVIEW_FIELDS,
    SKILL_FREQUENCY_FIELDS,
    TOP_SKILL_REVIEW_FIELDS,
    ZERO_SKILL_REVIEW_FIELDS,
)

REPORT_DIR = Path("reports/enrichment/run_75")


def test_generated_audit_csv_schemas_and_blank_annotations() -> None:
    expectations = {
        "top_skill_review.csv": (
            TOP_SKILL_REVIEW_FIELDS,
            ["expected_skills", "false_positive", "reviewer_notes"],
        ),
        "zero_skill_review.csv": (
            ZERO_SKILL_REVIEW_FIELDS,
            ["expected_skills", "missed_skill", "ontology_gap", "reviewer_notes"],
        ),
        "high_skill_count_review.csv": (
            HIGH_SKILL_COUNT_REVIEW_FIELDS,
            ["expected_skills", "false_positive_skills", "reviewer_notes"],
        ),
        "ambiguous_alias_review.csv": (
            AMBIGUOUS_ALIAS_REVIEW_FIELDS,
            ["expected_skills", "false_positive", "reviewer_notes"],
        ),
        "skill_frequency.csv": (SKILL_FREQUENCY_FIELDS, []),
        "evidence_integrity_violations.csv": (EVIDENCE_INTEGRITY_FIELDS, []),
    }

    for filename, (fieldnames, annotation_fields) in expectations.items():
        with (REPORT_DIR / filename).open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert reader.fieldnames == fieldnames
        assert all(row[field] == "" for row in rows for field in annotation_fields)


def test_generated_top_skill_review_is_balanced_and_includes_ai() -> None:
    with (REPORT_DIR / "top_skill_review.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["canonical"]] = counts.get(row["canonical"], 0) + 1

    assert len(rows) == 100
    assert set(counts.values()) == {10}
    assert "Artificial Intelligence" in counts


def test_generated_integrity_export_has_no_violations() -> None:
    with (REPORT_DIR / "evidence_integrity_violations.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert rows == []
