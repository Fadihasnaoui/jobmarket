"""Table-driven tests for normalization helpers."""

from __future__ import annotations

import pytest

from jobmarket.parse.normalize import (
    clean_text,
    fix_mojibake,
    normalize_company_name,
    normalize_salary,
    normalize_title,
    strip_html,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Python3 Developer", "python3 developer"),
        ("PYTHON", "python"),
        ("python 3.11", "python 3 11"),
        ("Senior Data Scientist (m/f/d)", "data scientist"),
        ("Jr. Backend Engineer", "backend engineer"),
        ("Sr. Cloud Architect", "cloud architect"),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Acme SARL", "acme"),
        ("ACME Sarl.", "acme"),
        ("Example Corp Ltd", "example"),
        ("Tech Solutions GmbH", "tech solutions"),
        ("Global Systems Inc.", "global systems"),
    ],
)
def test_normalize_company_name(raw: str, expected: str) -> None:
    assert normalize_company_name(raw) == expected


def test_strip_html_preserves_paragraphs() -> None:
    html = "<p>First paragraph.</p><p>Second paragraph.</p>"
    assert strip_html(html) == "First paragraph.\n\nSecond paragraph."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("fÃºtbol con pasiÃ³n", "fútbol con pasión"),
        ("Analista de CobranÃ§a Junior", "Analista de Cobrança Junior"),
        ("SuperlÃ³gica Tecnologias", "Superlógica Tecnologias"),
        ("Développeur Full Stack", "Développeur Full Stack"),
        ("H/F poste basé à Paris", "H/F poste basé à Paris"),
        ("", ""),
    ],
)
def test_fix_mojibake(raw: str, expected: str) -> None:
    assert fix_mojibake(raw) == expected


def test_strip_html_repairs_mojibake_in_description() -> None:
    html = "<p>fÃºtbol con pasiÃ³n</p>"
    assert strip_html(html) == "fútbol con pasión"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  SuperlÃ³gica Tecnologias  ", "Superlógica Tecnologias"),
        ("   ", None),
        (None, None),
    ],
)
def test_clean_text(raw: str | None, expected: str | None) -> None:
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("salary_min", "salary_max", "expected_min", "expected_max"),
    [
        (50000, None, 50000.0, 50000.0),
        (None, 80000, 80000.0, 80000.0),
        (40000, 60000, 40000.0, 60000.0),
    ],
)
def test_normalize_salary(
    salary_min: int | None,
    salary_max: int | None,
    expected_min: float,
    expected_max: float,
) -> None:
    min_val, max_val, currency, period = normalize_salary(
        salary_min,
        salary_max,
        currency="eur",
        period="Year",
    )
    assert min_val == expected_min
    assert max_val == expected_max
    assert currency == "EUR"
    assert period == "year"
