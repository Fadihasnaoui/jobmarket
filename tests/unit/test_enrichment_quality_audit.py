"""Tests for deterministic matcher quality-audit helpers."""

from __future__ import annotations

from types import SimpleNamespace

from jobmarket.reporting.enrichment_quality import (
    AMBIGUOUS_ALIAS_REVIEW_FIELDS,
    HIGH_SKILL_COUNT_REVIEW_FIELDS,
    TOP_SKILL_REVIEW_FIELDS,
    ZERO_SKILL_REVIEW_FIELDS,
    _canonical_job_sort_key,
    _violation,
    context_around,
    description_excerpt,
    deterministic_sample_ids,
    is_ambiguous_alias,
)
from jobmarket.skills.ontology import normalize_alias


def test_deterministic_sample_ids_is_stable_and_sorted() -> None:
    ids = list(range(1, 30))

    assert deterministic_sample_ids(ids, 8, 123) == deterministic_sample_ids(ids, 8, 123)
    assert deterministic_sample_ids(list(reversed(ids)), 8, 123) == deterministic_sample_ids(
        ids, 8, 123
    )
    assert deterministic_sample_ids([3, 1, 2], 10, 123) == [1, 2, 3]


def test_review_csv_schemas_include_empty_annotation_columns() -> None:
    assert TOP_SKILL_REVIEW_FIELDS[-3:] == [
        "expected_skills",
        "false_positive",
        "reviewer_notes",
    ]
    assert ZERO_SKILL_REVIEW_FIELDS[-4:] == [
        "expected_skills",
        "missed_skill",
        "ontology_gap",
        "reviewer_notes",
    ]
    assert HIGH_SKILL_COUNT_REVIEW_FIELDS[-3:] == [
        "expected_skills",
        "false_positive_skills",
        "reviewer_notes",
    ]
    assert AMBIGUOUS_ALIAS_REVIEW_FIELDS[-3:] == [
        "expected_skills",
        "false_positive",
        "reviewer_notes",
    ]


def test_ambiguous_alias_heuristic_catches_short_punctuation_and_common_words() -> None:
    assert is_ambiguous_alias("AI") is True
    assert is_ambiguous_alias("C++") is True
    assert is_ambiguous_alias("chef") is True
    assert is_ambiguous_alias("Python") is False


def test_excerpt_and_context_are_deterministic() -> None:
    description = "  This   is a long description with Python and SQL.  "

    assert description_excerpt(description, limit=20) == "This is a long descr"
    assert context_around("abc Python xyz", 4, 10, size=3) == ("bc ", " xy")


def test_canonical_job_sort_key_and_violation_shape() -> None:
    assert _canonical_job_sort_key({"canonical": "Python", "job_id": 7}) == ("Python", 7)

    violation = _violation(75, 10, 20, "Python", "bad", "details")
    assert list(violation) == [
        "run_id",
        "job_id",
        "job_skill_id",
        "canonical",
        "violation_type",
        "details",
    ]
    assert violation["violation_type"] == "bad"


def test_alias_normalization_accepts_case_and_accents_for_integrity_rule() -> None:
    row = SimpleNamespace(matched_alias="intelligence artificielle")
    assert normalize_alias(row.matched_alias) == normalize_alias("Intelligence Artificielle")
