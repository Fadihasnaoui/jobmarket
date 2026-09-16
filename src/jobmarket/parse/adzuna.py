"""Adzuna payload parser."""

from __future__ import annotations

from datetime import UTC, datetime

from jobmarket.parse.base import AbstractParser, ParsedJob
from jobmarket.parse.normalize import normalize_salary, strip_html

_COUNTRY_TO_ISO2: dict[str, str] = {
    "uk": "gb",
    "united kingdom": "gb",
    "gb": "gb",
    "france": "fr",
    "fr": "fr",
    "germany": "de",
    "de": "de",
    "deutschland": "de",
    "netherlands": "nl",
    "nederland": "nl",
    "nl": "nl",
    "spain": "es",
    "españa": "es",
    "espana": "es",
    "es": "es",
    "italy": "it",
    "italia": "it",
    "it": "it",
    "ireland": "ie",
    "ie": "ie",
    "belgium": "be",
    "be": "be",
    "switzerland": "ch",
    "ch": "ch",
    "austria": "at",
    "at": "at",
    "usa": "us",
    "us": "us",
    "united states": "us",
}

_CONTRACT_MAP: dict[str, str] = {
    "full_time": "full_time",
    "part_time": "part_time",
    "permanent": "full_time",
    "contract": "contract",
    "internship": "internship",
    "temporary": "contract",
}


class AdzunaParser(AbstractParser):
    """Parse Adzuna search result payloads."""

    source = "adzuna"

    def parse(self, payload: dict[str, object]) -> ParsedJob:
        title = _require_str(payload, "title")
        description_html = _optional_str(payload, "description") or ""
        company = _optional_dict(payload, "company")
        # Adzuna sometimes anonymizes the employer (agency-confidential listings),
        # returning a company object with no display_name at all. Same "Unknown"
        # fallback as JoobleParser, rather than dropping otherwise-good listings.
        company_name = (_optional_str(company, "display_name") if company else None) or "Unknown"

        location = _optional_dict(payload, "location")
        location_raw = _optional_str(location, "display_name") if location else None
        country, city = _parse_location_area(location)

        salary_min = _optional_number(payload, "salary_min")
        salary_max = _optional_number(payload, "salary_max")
        salary_min, salary_max, salary_currency, salary_period = normalize_salary(
            salary_min,
            salary_max,
            currency=None,
            period="year" if salary_min is not None or salary_max is not None else None,
        )

        contract_type = _parse_contract_type(payload)
        posted_at = _parse_posted_at(payload)
        url = _optional_str(payload, "redirect_url")

        return ParsedJob(
            company_name=company_name,
            title=title,
            description=strip_html(description_html),
            location_raw=location_raw,
            country=country,
            city=city,
            is_remote=None,
            contract_type=contract_type,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_is_predicted=_parse_salary_is_predicted(payload),
            posted_at=posted_at,
            url=url,
        )


def _parse_salary_is_predicted(payload: dict[str, object]) -> bool:
    value = payload.get("salary_is_predicted")
    if value is None:
        return False
    return str(value) == "1"


def _parse_contract_type(payload: dict[str, object]) -> str | None:
    contract_time = _optional_str(payload, "contract_time")
    if contract_time:
        return _CONTRACT_MAP.get(contract_time.lower())

    contract_type = _optional_str(payload, "contract_type")
    if contract_type:
        return _CONTRACT_MAP.get(contract_type.lower())

    return None


def _parse_posted_at(payload: dict[str, object]) -> datetime | None:
    created = _optional_str(payload, "created")
    if not created:
        return None
    normalized = created.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _parse_location_area(location: dict[str, object] | None) -> tuple[str | None, str | None]:
    if location is None:
        return None, None

    area = location.get("area")
    if not isinstance(area, list) or not area:
        return None, None

    country = _to_iso2(str(area[0]))
    city = str(area[1]) if len(area) > 1 else None
    return country, city


def _to_iso2(value: str) -> str | None:
    key = value.strip().casefold()
    mapped = _COUNTRY_TO_ISO2.get(key)
    if mapped is not None:
        return mapped
    # Fallback only for 2-letter tokens the dict doesn't yet know about — the dict
    # must be checked first, since some non-ISO2 country names are ALSO exactly two
    # letters (e.g. Adzuna returns "UK", which must map to "gb", not pass through
    # as-is).
    if len(key) == 2 and key.isalpha():
        return key
    return None


def _require_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {key}")
    return value.strip()


def _optional_str(data: dict[str, object] | None, key: str) -> str | None:
    if data is None:
        return None
    value = data.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _optional_dict(data: dict[str, object], key: str) -> dict[str, object] | None:
    value = data.get(key)
    if isinstance(value, dict):
        return value
    return None


def _optional_number(data: dict[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None
