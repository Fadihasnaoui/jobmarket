"""Tests for metadata-only manual gold selection proposals."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from jobmarket.datasets.gold_selection import generate_manual_gold_selection_proposal


def test_gold_selection_proposal_is_balanced_private_and_deterministic(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    shortlist_path = reports / "gold_candidate_shortlist.csv"
    template_path = reports / "manual_gold_selection_template.csv"
    audit_path = reports / "dataset_audit.json"
    duplicates_path = reports / "duplicates.json"
    registry_path = tmp_path / "registry.json"
    _write_fixture(shortlist_path, template_path, audit_path, duplicates_path, registry_path)
    template_before = template_path.read_text(encoding="utf-8")

    summary = generate_manual_gold_selection_proposal(
        shortlist_path=shortlist_path,
        template_path=template_path,
        audit_path=audit_path,
        duplicates_path=duplicates_path,
        registry_path=registry_path,
        reports_dir=reports,
        selected_total=6,
        split_targets={"development": 3, "validation": 2, "test": 1},
    )

    rows = _read_csv(summary.output_csv)
    selected = [row for row in rows if row["decision"] == "proposed_accept_private_only"]
    rejected = [row for row in rows if row["decision"] == "proposed_reject"]
    assert summary.selected_count == 6
    assert len(selected) == 6
    assert len(rejected) == 4
    assert summary.split_counts == {"development": 3, "test": 1, "validation": 2}
    assert Counter(row["assigned_split"] for row in selected) == Counter(
        {"development": 3, "validation": 2, "test": 1}
    )
    assert len({row["selected_gold_id"] for row in selected}) == 6
    assert all(row["selected_gold_id"].startswith("cv_gold_") for row in selected)
    assert all(row["assigned_split"] for row in selected)
    assert all(not row["selected_gold_id"] and not row["assigned_split"] for row in rejected)
    assert {row["decision"] for row in rows} <= {
        "proposed_accept_private_only",
        "proposed_reject",
        "manual_review_required",
    }
    assert all(row["licence_review_complete"] == "no" for row in rows)
    assert all(row["pii_review_complete"] == "no" for row in rows)
    assert all(row["annotation_complete"] == "no" for row in rows)
    assert "GC-010" not in {row["shortlist_id"] for row in selected}
    assert template_path.read_text(encoding="utf-8") == template_before


def test_gold_selection_outputs_privacy_safe_report_and_exceptions(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    shortlist_path = reports / "gold_candidate_shortlist.csv"
    template_path = reports / "manual_gold_selection_template.csv"
    audit_path = reports / "dataset_audit.json"
    duplicates_path = reports / "duplicates.json"
    registry_path = tmp_path / "registry.json"
    _write_fixture(shortlist_path, template_path, audit_path, duplicates_path, registry_path)

    summary = generate_manual_gold_selection_proposal(
        shortlist_path=shortlist_path,
        template_path=template_path,
        audit_path=audit_path,
        duplicates_path=duplicates_path,
        registry_path=registry_path,
        reports_dir=reports,
        selected_total=6,
        split_targets={"development": 3, "validation": 2, "test": 1},
    )

    markdown = summary.output_markdown.read_text(encoding="utf-8")
    proposed = summary.output_csv.read_text(encoding="utf-8")
    exceptions = _read_csv(summary.exceptions_csv)
    assert "person@example.com" not in markdown
    assert "555-0100" not in markdown
    assert "Raw private resume text" not in markdown
    assert "person@example.com" not in proposed
    assert exceptions == []


def _write_fixture(
    shortlist_path: Path,
    template_path: Path,
    audit_path: Path,
    duplicates_path: Path,
    registry_path: Path,
) -> None:
    domains = [
        (
            "GC-001",
            "Data / AI",
            "resume_pdf",
            "Data Science",
            True,
            "UNKNOWN",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-002",
            "Software Engineering",
            "resume_pdf",
            "SAP Developer",
            True,
            "UNKNOWN",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-003",
            "Management",
            "resume_classification",
            "CONSULTANT",
            False,
            "en",
            "email=0; phone=1; url=0; address_like=0",
        ),
        (
            "GC-004",
            "Sales",
            "resume_classification",
            "SALES",
            False,
            "en",
            "email=1; phone=0; url=0; address_like=0",
        ),
        (
            "GC-005",
            "Finance",
            "resume_classification",
            "FINANCE",
            False,
            "en",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-006",
            "Accounting",
            "resume_classification",
            "ACCOUNTANT",
            False,
            "en",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-007",
            "Healthcare",
            "resume_pdf",
            "HEALTHCARE",
            True,
            "UNKNOWN",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-008",
            "Education",
            "resume_classification",
            "EDUCATION",
            False,
            "en",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-009",
            "Human Resources",
            "resume_pdf",
            "HR",
            True,
            "UNKNOWN",
            "email=0; phone=0; url=0; address_like=0",
        ),
        (
            "GC-010",
            "Finance",
            "resume_classification",
            "FINANCE",
            False,
            "en",
            "email=0; phone=0; url=0; address_like=0",
        ),
    ]
    shortlist_rows = []
    audit_rows = []
    for index, (sid, domain, source, category, ocr, lang, pii) in enumerate(domains, start=1):
        rel = f"{source}/{category}/candidate_{index}.pdf"
        shortlist_rows.append(
            {
                "shortlist_id": sid,
                "source_dataset": source,
                "relative_path": rel,
                "format": "PDF",
                "file_size": "1000",
                "page_count": "" if ocr else "2",
                "text_layer_available": str(not ocr),
                "ocr_likely_required": str(ocr),
                "detected_language": lang,
                "text_quality": "0.0" if ocr else "1.0",
                "detected_category": category,
                "duplicate_status": "unique",
                "pii_indicators": pii,
                "recommended_gold_role": f"{domain}; multi-page CV"
                if not ocr
                else f"{domain}; poor-CV abstention cases",
                "review_priority": "high",
                "reason_for_inclusion": f"Covers {domain}; safe metadata only",
                "licence_status": "TO_VERIFY",
                "redistribution_status": "NOT_APPROVED_PENDING_LICENSE_AND_PII_REVIEW",
            }
        )
        audit_rows.append({"relative_path": rel, "warnings": ["EmptyDocumentError"] if ocr else []})
    _write_csv(shortlist_path, list(shortlist_rows[0]), shortlist_rows)
    _write_csv(
        template_path,
        [
            "shortlist_id",
            "decision",
            "decision_reason",
            "selected_gold_id",
            "needs_anonymization",
            "licence_review_complete",
            "pii_review_complete",
            "annotation_complete",
            "assigned_split",
            "reviewer_notes",
        ],
        [{"shortlist_id": row["shortlist_id"]} for row in shortlist_rows],
    )
    audit_path.write_text(json.dumps({"documents": audit_rows}), encoding="utf-8")
    duplicate_path = shortlist_rows[-1]["relative_path"]
    duplicates_path.write_text(
        json.dumps(
            {
                "sha256_duplicates": [],
                "normalized_text_duplicates": [{"files": [duplicate_path]}],
                "near_duplicate_groups": [{"files": [duplicate_path]}],
            }
        ),
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "resume_pdf", "license_status": "TO_VERIFY"},
                    {"name": "resume_classification", "license_status": "TO_VERIFY"},
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
