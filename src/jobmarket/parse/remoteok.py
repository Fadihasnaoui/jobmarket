"""RemoteOK payload parser."""

from __future__ import annotations

from jobmarket.parse.base import AbstractParser, ParsedJob
from jobmarket.parse.helpers import (
    optional_number,
    optional_str,
    parse_iso_datetime,
    parse_unix_epoch,
)
from jobmarket.parse.normalize import clean_text, normalize_salary, strip_html


class RemoteOKParser(AbstractParser):
    """Parse RemoteOK API job payloads."""

    source = "remoteok"

    def parse(self, payload: dict[str, object]) -> ParsedJob:
        title = clean_text(optional_str(payload, "position") or optional_str(payload, "title"))
        if not title:
            raise ValueError("Missing job title")

        company_name = clean_text(optional_str(payload, "company"))
        if not company_name:
            raise ValueError("Missing company name")

        description = strip_html(optional_str(payload, "description") or "")
        location_raw = clean_text(optional_str(payload, "location"))

        salary_min = optional_number(payload, "salary_min")
        salary_max = optional_number(payload, "salary_max")
        salary_min, salary_max, salary_currency, salary_period = normalize_salary(
            salary_min,
            salary_max,
            currency="USD" if salary_min is not None or salary_max is not None else None,
            period="year" if salary_min is not None or salary_max is not None else None,
        )

        posted_at = parse_unix_epoch(payload.get("epoch"))
        if posted_at is None:
            posted_at = parse_iso_datetime(optional_str(payload, "date"))

        url = optional_str(payload, "url") or optional_str(payload, "apply_url")

        return ParsedJob(
            company_name=company_name,
            title=title,
            description=description,
            location_raw=location_raw,
            country=None,
            city=None,
            is_remote=True,
            contract_type=None,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_is_predicted=False,
            posted_at=posted_at,
            url=url,
        )
