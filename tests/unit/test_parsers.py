"""Parser unit tests against saved fixtures."""

from __future__ import annotations

import pytest

from jobmarket.parse.adzuna import AdzunaParser
from jobmarket.parse.jooble import JoobleParser
from jobmarket.parse.normalize import normalize_company_name, normalize_title
from jobmarket.parse.remoteok import RemoteOKParser


def test_adzuna_parser(adzuna_payload: dict[str, object]) -> None:
    parsed = AdzunaParser().parse(adzuna_payload)

    assert parsed.title == "Senior Data Scientist (m/f/d)"
    assert parsed.company_name == "Acme Analytics SARL"
    assert parsed.country == "fr"
    assert parsed.city == "Paris"
    assert parsed.salary_min == 50000.0
    assert parsed.salary_max == 65000.0
    assert parsed.salary_is_predicted is True
    assert parsed.contract_type == "full_time"
    assert "Build production ML models." in parsed.description
    assert parsed.url == "https://example.com/jobs/adzuna-4829103746"


def test_remoteok_parser(remoteok_payload: dict[str, object]) -> None:
    parsed = RemoteOKParser().parse(remoteok_payload)

    assert parsed.title == "Backend Developer"
    assert parsed.company_name == "Example Corp Ltd"
    assert parsed.is_remote is True
    assert parsed.salary_min == 80000.0
    assert parsed.salary_max == 120000.0
    assert parsed.salary_currency == "USD"
    assert parsed.posted_at is not None
    assert "Build APIs with Python" in parsed.description


def test_jooble_parser(jooble_payload: dict[str, object]) -> None:
    parsed = JoobleParser().parse(jooble_payload)

    assert parsed.title == "Machine Learning Engineer"
    assert parsed.company_name == "Tech Solutions GmbH"
    assert parsed.country == "de"
    assert parsed.city == "Berlin"
    assert parsed.salary_min == 70000.0
    assert parsed.salary_max == 90000.0
    assert parsed.salary_currency == "EUR"
    assert parsed.contract_type == "full_time"
    assert parsed.url == "https://example.com/jobs/ml-engineer-9001"


@pytest.mark.parametrize(
    ("location", "expected_country"),
    [
        ("Berlin, Germany", "de"),
        ("London, UK", "gb"),  # regression: 2-letter shortcut used to bypass the dict
        ("Amsterdam, Nederland", "nl"),  # regression: missing Dutch spelling
        ("Rome, Italia", "it"),
        ("Paris, GB", "gb"),
        ("Italian Republic", "it"),
        ("Kingdom of Spain", "es"),
    ],
)
def test_jooble_country_mapping(
    jooble_payload: dict[str, object], location: str, expected_country: str
) -> None:
    payload = {**jooble_payload, "location": location}
    parsed = JoobleParser().parse(payload)
    assert parsed.country == expected_country


def test_normalized_fields_match_fixture_titles(adzuna_payload: dict[str, object]) -> None:
    parsed = AdzunaParser().parse(adzuna_payload)
    assert normalize_title(parsed.title) == "data scientist"
    assert normalize_company_name(parsed.company_name) == "acme analytics"


@pytest.mark.parametrize(
    ("area", "expected_country"),
    [
        (["France", "Paris"], "fr"),
        (["UK", "London"], "gb"),  # regression: 2-letter shortcut used to bypass the dict
        (["Nederland", "Amsterdam"], "nl"),  # regression: missing Dutch spelling
        (["Deutschland", "Berlin"], "de"),
        (["Italia", "Roma"], "it"),
        (["GB"], "gb"),
        (["XX"], "xx"),  # unmapped 2-letter token still passes through as a fallback
    ],
)
def test_adzuna_country_mapping(
    adzuna_payload: dict[str, object], area: list[str], expected_country: str
) -> None:
    payload = {**adzuna_payload, "location": {**adzuna_payload["location"], "area": area}}
    parsed = AdzunaParser().parse(payload)
    assert parsed.country == expected_country


def test_adzuna_parser_falls_back_to_unknown_company(
    adzuna_payload: dict[str, object],
) -> None:
    """Adzuna anonymizes some listings (agency-confidential): company object with no
    display_name at all. This is real, recoverable data, not malformed input — same
    "Unknown" fallback as JoobleParser, not a dropped row."""
    payload = {**adzuna_payload, "company": {"__CLASS__": "Adzuna::API::Response::Company"}}
    parsed = AdzunaParser().parse(payload)
    assert parsed.company_name == "Unknown"


def test_remoteok_parser_repairs_mojibake() -> None:
    payload = {
        "id": "900002",
        "company": "SuperlÃ³gica Tecnologias",
        "position": "Analista CobranÃ§a Jr Remoto",
        "description": "<p>fÃºtbol con pasiÃ³n</p>",
        "location": "Brazil",
        "epoch": "1710000000",
    }
    parsed = RemoteOKParser().parse(payload)

    assert parsed.company_name == "Superlógica Tecnologias"
    assert parsed.title == "Analista Cobrança Jr Remoto"
    assert parsed.description == "fútbol con pasión"
