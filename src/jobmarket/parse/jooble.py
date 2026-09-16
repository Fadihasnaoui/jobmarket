"""Jooble payload parser."""

from __future__ import annotations

import re

from jobmarket.parse.base import AbstractParser, ParsedJob
from jobmarket.parse.helpers import optional_str, parse_iso_datetime, require_str
from jobmarket.parse.normalize import normalize_salary, strip_html

_LOCATION_TO_COUNTRY: dict[str, str] = {
    "france": "fr",
    "germany": "de",
    "netherlands": "nl",
    "spain": "es",
    "deutschland": "de",
    "nederland": "nl",
    "españa": "es",
    "espana": "es",
    "united kingdom": "gb",
    "uk": "gb",
    "great britain": "gb",
    "italy": "it",
    "italia": "it",
    "italian republic": "it",
    "kingdom of spain": "es",
}

_CONTRACT_MAP: dict[str, str] = {
    "full-time": "full_time",
    "full time": "full_time",
    "part-time": "part_time",
    "part time": "part_time",
    "contract": "contract",
    "internship": "internship",
    "temporary": "contract",
}

_SALARY_RANGE_PATTERN = re.compile(
    r"(?P<min>[\d][\d,\.]*)"
    r"(?:\s*[-–]\s*(?P<max>[\d][\d,\.]*))?"
)


class JoobleParser(AbstractParser):
    """Parse Jooble search result payloads."""

    source = "jooble"

    def parse(self, payload: dict[str, object]) -> ParsedJob:
        title = require_str(payload, "title")
        company_name = optional_str(payload, "company") or "Unknown"
        snippet = optional_str(payload, "snippet") or ""
        description = strip_html(snippet)

        location_raw = optional_str(payload, "location")
        country, city = _parse_location(location_raw)

        salary_min, salary_max, salary_currency, salary_period = _parse_salary(
            optional_str(payload, "salary")
        )

        contract_type = _parse_contract_type(optional_str(payload, "type"))
        posted_at = parse_iso_datetime(optional_str(payload, "updated"))
        url = optional_str(payload, "link")

        return ParsedJob(
            company_name=company_name,
            title=title,
            description=description,
            location_raw=location_raw,
            country=country,
            city=city,
            is_remote=None,
            contract_type=contract_type,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_is_predicted=False,
            posted_at=posted_at,
            url=url,
        )


def _parse_location(location_raw: str | None) -> tuple[str | None, str | None]:
    if not location_raw:
        return None, None

    parts = [part.strip() for part in location_raw.split(",") if part.strip()]
    if not parts:
        return None, None

    country = _location_token_to_country(parts[-1])
    city = parts[0] if len(parts) > 1 else None
    return country, city


def _location_token_to_country(token: str) -> str | None:
    key = token.casefold()
    mapped = _LOCATION_TO_COUNTRY.get(key)
    if mapped is not None:
        return mapped
    # Fallback only for 2-letter tokens the dict doesn't yet know about — check the
    # dict first, since some non-ISO2 names are also exactly two letters (e.g. "UK"
    # must map to "gb", not pass through as-is).
    if len(key) == 2 and key.isalpha():
        return key
    return None


def _parse_contract_type(job_type: str | None) -> str | None:
    if not job_type:
        return None
    return _CONTRACT_MAP.get(job_type.casefold())


def _parse_salary(
    salary_raw: str | None,
) -> tuple[float | None, float | None, str | None, str | None]:
    if not salary_raw:
        return None, None, None, None

    currency = None
    if "$" in salary_raw:
        currency = "USD"
    elif "€" in salary_raw:
        currency = "EUR"
    elif "£" in salary_raw:
        currency = "GBP"

    match = _SALARY_RANGE_PATTERN.search(re.sub(r"[^\d,\.\-–]", "", salary_raw))
    if not match:
        return None, None, currency, None

    salary_min = float(match.group("min").replace(",", ""))
    max_group = match.group("max")
    salary_max = float(max_group.replace(",", "")) if max_group else salary_min
    return normalize_salary(
        salary_min,
        salary_max,
        currency=currency,
        period="year",
    )
