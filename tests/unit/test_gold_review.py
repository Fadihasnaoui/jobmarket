"""Tests for privacy-safe manual gold review package."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from jobmarket.datasets.gold_review import (
    format_review_item,
    generate_manual_review_package,
    generate_prefilled_final_review,
    review_progress,
    safe_review_item,
    validate_final_review,
)


def test_manual_review_package_groups_licence_pii_and_template(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)

    result = generate_manual_review_package(**paths)

    assert result["selected_count"] == 4
    source_rows = _read_csv(tmp_path / "reports" / "gold_review_by_source.csv")
    licence_rows = _read_csv(tmp_path / "reports" / "licence_review_checklist.csv")
    pii_rows = _read_csv(tmp_path / "reports" / "pii_review_queue.csv")
    final_rows = _read_csv(tmp_path / "reports" / "manual_gold_final_review.csv")
    assert {row["source_dataset"] for row in source_rows} == {"resume_pdf", "resume_classification"}
    assert len(licence_rows) == 2
    assert {row["registry_licence_status"] for row in licence_rows} == {"TO_VERIFY"}
    assert all(row["final_manual_decision"] == "" for row in licence_rows)
    assert all(row["reviewer_notes"] == "" for row in licence_rows)
    assert {row["recommended_action"] for row in pii_rows} >= {
        "REVIEW_AND_ANONYMIZE",
        "REVIEW_UNCERTAIN",
    }
    assert all("person@example.com" not in str(row) for row in pii_rows)
    assert all(row["final_decision"] == "" for row in final_rows)
    assert all(row["licence_review_complete"] == "" for row in final_rows)


def test_review_item_by_shortlist_and_gold_id_is_privacy_safe(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)

    by_shortlist = safe_review_item(
        shortlist_id="GC-001",
        proposed_path=paths["proposed_path"],
        shortlist_path=paths["shortlist_path"],
        audit_path=paths["audit_path"],
    )
    by_gold = safe_review_item(
        gold_id="cv_gold_001",
        proposed_path=paths["proposed_path"],
        shortlist_path=paths["shortlist_path"],
        audit_path=paths["audit_path"],
    )
    output = format_review_item(by_shortlist)
    assert by_shortlist == by_gold
    assert "person@example.com" not in output
    assert "555-0100" not in output
    assert "Raw private resume text" not in output
    assert "pii_indicators" in output


def test_validate_final_review_rejects_unresolved_licence_pii_and_anonymization(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)
    review = tmp_path / "reports" / "manual_gold_final_review.csv"
    rows = _read_csv(review)
    rows[0].update(
        {
            "final_decision": "accept",
            "final_split": "development",
            "licence_review_complete": "no",
            "licence_decision": "allow_local_gold_use",
            "pii_review_complete": "no",
            "needs_anonymization": "yes",
            "anonymization_complete": "no",
        }
    )
    _write_csv(review, list(rows[0]), rows)

    result = validate_final_review(
        review_path=review,
        proposed_path=paths["proposed_path"],
        shortlist_path=paths["shortlist_path"],
        duplicates_path=paths["duplicates_path"],
        reports_dir=tmp_path / "reports",
    )

    errors = {error["error"] for error in result["errors"]}
    assert "accepted_rows_need_licence_review_complete" in errors
    assert "accepted_rows_need_pii_review_complete" in errors
    assert "anonymization_required_before_acceptance" in errors
    assert result["valid"] is False


def test_validate_final_review_private_only_and_duplicate_split_protection(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)
    review = tmp_path / "reports" / "manual_gold_final_review.csv"
    rows = _read_csv(review)
    for index, row in enumerate(rows):
        row.update(
            {
                "final_decision": "accept_private_only",
                "final_split": "development" if index == 0 else "validation",
                "licence_review_complete": "yes",
                "licence_decision": "private_only",
                "pii_review_complete": "yes",
                "needs_anonymization": "no",
                "anonymization_complete": "",
                "approved_private_only": "no" if index == 0 else "yes",
            }
        )
    _write_csv(review, list(rows[0]), rows)

    result = validate_final_review(
        review_path=review,
        proposed_path=paths["proposed_path"],
        shortlist_path=paths["shortlist_path"],
        duplicates_path=paths["duplicates_path"],
        reports_dir=tmp_path / "reports",
    )

    errors = {error["error"] for error in result["errors"]}
    assert "private_only_acceptance_requires_approval" in errors
    assert "duplicate_or_near_duplicate_crosses_splits" in errors
    assert result["valid"] is False



def test_prefilled_final_review_preserves_original_and_keeps_approvals_blank(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)
    original = tmp_path / "reports" / "manual_gold_final_review.csv"
    original_before = original.read_text(encoding="utf-8")

    result = generate_prefilled_final_review(
        final_review_path=original,
        pii_queue_path=tmp_path / "reports" / "pii_review_queue.csv",
        reports_dir=tmp_path / "reports",
    )

    assert original.read_text(encoding="utf-8") == original_before
    rows = _read_csv(tmp_path / "reports" / "manual_gold_final_review_prefilled.csv")
    assert result["rows_prefilled"] == 4
    assert all(row["final_decision"] == "" for row in rows)
    assert all(row["final_split"] == row["proposed_split"] for row in rows)
    assert all(row["licence_review_complete"] == "no" for row in rows)
    assert all(row["licence_decision"] == "TO_VERIFY" for row in rows)
    assert all(row["pii_review_complete"] == "no" for row in rows)
    assert all(row["anonymization_complete"] == "no" for row in rows)
    assert all(row["approved_private_only"] == "no" for row in rows)
    assert all(row["annotation_complete"] == "no" for row in rows)
    assert {row["needs_anonymization"] for row in rows} == {"yes", "review_required"}


def test_prefilled_reports_are_privacy_safe_and_action_queue_prioritized(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)

    generate_prefilled_final_review(
        final_review_path=tmp_path / "reports" / "manual_gold_final_review.csv",
        pii_queue_path=tmp_path / "reports" / "pii_review_queue.csv",
        reports_dir=tmp_path / "reports",
    )

    markdown = (tmp_path / "reports" / "manual_gold_final_review_prefilled.md").read_text(
        encoding="utf-8"
    )
    queue = _read_csv(tmp_path / "reports" / "manual_gold_review_action_queue.csv")
    assert "person@example.com" not in markdown
    assert "555-0100" not in markdown
    assert "Raw private resume text" not in markdown
    assert queue[0]["priority"] == "1"
    assert queue[0]["completion_status"] == "PENDING"
    assert queue[0]["safe_document_review_command"].startswith(
        "python -m jobmarket dataset review-item --shortlist-id"
    )


def test_review_progress_counts_blocked_prefilled_rows(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)
    generate_prefilled_final_review(
        final_review_path=tmp_path / "reports" / "manual_gold_final_review.csv",
        pii_queue_path=tmp_path / "reports" / "pii_review_queue.csv",
        reports_dir=tmp_path / "reports",
    )

    progress = review_progress(tmp_path / "reports" / "manual_gold_final_review_prefilled.csv")

    assert progress["total_rows"] == 4
    assert progress["rows_with_final_decision"] == 0
    assert progress["licence_reviews_completed"] == 0
    assert progress["pii_reviews_completed"] == 0
    assert progress["rows_ready_for_validation"] == 0
    assert progress["rows_blocked_by_licence"] == 4
    assert progress["rows_blocked_by_pii"] == 4
    assert progress["rows_blocked_by_anonymization"] == 3
    assert progress["ready_split_counts"] == {}


def test_validate_final_review_has_no_promotion_side_effects(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    generate_manual_review_package(**paths)
    review = tmp_path / "reports" / "manual_gold_final_review.csv"

    validate_final_review(
        review_path=review,
        proposed_path=paths["proposed_path"],
        shortlist_path=paths["shortlist_path"],
        duplicates_path=paths["duplicates_path"],
        reports_dir=tmp_path / "reports",
    )

    assert not (tmp_path / "gold" / "documents").exists()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    reports = tmp_path / "reports"
    raw = tmp_path / "raw"
    reports.mkdir()
    (raw / "resume_pdf").mkdir(parents=True)
    (raw / "resume_classification").mkdir(parents=True)
    shortlist_path = reports / "gold_candidate_shortlist.csv"
    proposed_path = reports / "manual_gold_selection_proposed.csv"
    audit_path = reports / "dataset_audit.json"
    duplicates_path = reports / "duplicates.json"
    registry_path = tmp_path / "registry.json"
    selected = [
        _shortlist(
            "GC-001",
            "cv_gold_001",
            "resume_pdf",
            "Data Science",
            True,
            "email=1; phone=0; url=0; address_like=0",
        ),
        _shortlist(
            "GC-002",
            "cv_gold_002",
            "resume_pdf",
            "Data Science",
            True,
            "email=0; phone=0; url=0; address_like=0",
        ),
        _shortlist(
            "GC-003",
            "cv_gold_003",
            "resume_classification",
            "FINANCE",
            False,
            "email=0; phone=1; url=0; address_like=0",
        ),
        _shortlist(
            "GC-004",
            "cv_gold_004",
            "resume_classification",
            "SALES",
            False,
            "email=0; phone=0; url=1; address_like=0",
        ),
    ]
    _write_csv(shortlist_path, list(selected[0][0]), [item[0] for item in selected])
    _write_csv(proposed_path, list(selected[0][1]), [item[1] for item in selected])
    audit_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"relative_path": item[0]["relative_path"], "warnings": ["EmptyDocumentError"]}
                    for item in selected
                ]
            }
        ),
        encoding="utf-8",
    )
    duplicates_path.write_text(
        json.dumps(
            {
                "sha256_duplicates": [],
                "normalized_text_duplicates": [
                    {"files": [selected[0][0]["relative_path"], selected[1][0]["relative_path"]]}
                ],
                "near_duplicate_groups": [],
            }
        ),
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "resume_pdf", "license": "UNKNOWN", "license_status": "TO_VERIFY"},
                    {
                        "name": "resume_classification",
                        "license": "UNKNOWN",
                        "license_status": "TO_VERIFY",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "proposed_path": proposed_path,
        "shortlist_path": shortlist_path,
        "registry_path": registry_path,
        "audit_path": audit_path,
        "reports_dir": reports,
        "raw_dir": raw,
        "duplicates_path": duplicates_path,
    }


def _shortlist(
    shortlist_id: str,
    gold_id: str,
    source: str,
    category: str,
    ocr: bool,
    pii: str,
) -> tuple[dict[str, str], dict[str, str]]:
    rel = f"{source}/{category}/{shortlist_id}.pdf"
    shortlist = {
        "shortlist_id": shortlist_id,
        "source_dataset": source,
        "relative_path": rel,
        "format": "PDF",
        "file_size": "1000",
        "page_count": "" if ocr else "2",
        "text_layer_available": str(not ocr),
        "ocr_likely_required": str(ocr),
        "detected_language": "UNKNOWN" if ocr else "en",
        "text_quality": "0.0" if ocr else "1.0",
        "detected_category": category,
        "duplicate_status": "unique",
        "pii_indicators": pii,
        "recommended_gold_role": category,
        "review_priority": "high",
        "reason_for_inclusion": "safe metadata only",
        "licence_status": "TO_VERIFY",
        "redistribution_status": "NOT_APPROVED_PENDING_LICENSE_AND_PII_REVIEW",
    }
    proposed = {
        "shortlist_id": shortlist_id,
        "decision": "proposed_accept_private_only",
        "decision_reason": "adds_safe_metadata_coverage",
        "selected_gold_id": gold_id,
        "needs_anonymization": "yes",
        "licence_review_complete": "no",
        "pii_review_complete": "no",
        "annotation_complete": "no",
        "assigned_split": "development",
        "reviewer_notes": "Requires licence and PII validation before promotion",
    }
    return shortlist, proposed


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
