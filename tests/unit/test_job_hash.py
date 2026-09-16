"""Tests for job_hash stability and dedup behavior."""

from __future__ import annotations

from jobmarket.parse.dedup import compute_job_hash
from jobmarket.parse.normalize import normalize_company_name, normalize_title


def test_job_hash_is_stable() -> None:
    title_norm = normalize_title("Senior Data Scientist (m/f/d)")
    company_norm = normalize_company_name("Acme SARL")
    first = compute_job_hash(title_norm, company_norm, "fr")
    second = compute_job_hash(title_norm, company_norm, "fr")
    assert first == second


def test_job_hash_changes_with_country() -> None:
    title_norm = normalize_title("Data Engineer")
    company_norm = normalize_company_name("Acme Analytics")
    fr_hash = compute_job_hash(title_norm, company_norm, "fr")
    de_hash = compute_job_hash(title_norm, company_norm, "de")
    assert fr_hash != de_hash


def test_job_hash_treats_missing_country_as_empty_string() -> None:
    title_norm = normalize_title("DevOps Engineer")
    company_norm = normalize_company_name("Example Corp")
    assert compute_job_hash(title_norm, company_norm, None) == compute_job_hash(
        title_norm, company_norm, ""
    )
