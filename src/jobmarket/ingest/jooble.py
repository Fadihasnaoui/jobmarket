"""Jooble job board ingest source."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

import structlog

from jobmarket.config import Settings, get_settings
from jobmarket.ingest.base import AbstractSource, RawRecord
from jobmarket.ingest.constants import JOOBLE_LOCATIONS, SEARCH_TERMS
from jobmarket.ingest.http import ResilientHttpClient

logger = structlog.get_logger(__name__)

BASE_URL = "https://jooble.org/api"


def derive_jooble_source_id(payload: dict[str, object]) -> str:
    """Return Jooble job id, or sha256(link) when the API omits a stable id."""
    job_id = payload.get("id")
    if job_id is not None and str(job_id).strip():
        return str(job_id)

    link = payload.get("link")
    if isinstance(link, str) and link.strip():
        # Jooble does not always return a stable ID; derive from link when absent.
        return hashlib.sha256(link.strip().encode("utf-8")).hexdigest()

    raise ValueError("Jooble job missing both id and link")


class JoobleSource(AbstractSource):
    """Fetch job listings from the Jooble API."""

    name = "jooble"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def fetch(self) -> AsyncIterator[RawRecord]:
        if not self._settings.jooble_api_key:
            raise ValueError("JOOBLE_API_KEY must be set")

        url = f"{BASE_URL}/{self._settings.jooble_api_key}"
        async with ResilientHttpClient(self.name) as client:
            for location in JOOBLE_LOCATIONS:
                for term in SEARCH_TERMS:
                    page = 1
                    while True:
                        body = {
                            "keywords": term,
                            "location": location,
                            "page": str(page),
                        }
                        response = await client.post(url, json=body)
                        if response is None:
                            logger.warning(
                                "jooble_page_skipped",
                                location=location,
                                term=term,
                                page=page,
                            )
                            break

                        try:
                            data = response.json()
                        except ValueError:
                            logger.warning(
                                "jooble_invalid_json",
                                location=location,
                                term=term,
                                page=page,
                            )
                            break

                        jobs = data.get("jobs", [])
                        if not isinstance(jobs, list) or not jobs:
                            break

                        for job in jobs:
                            if not isinstance(job, dict):
                                continue
                            try:
                                source_id = derive_jooble_source_id(job)
                            except ValueError:
                                logger.warning(
                                    "jooble_missing_identity",
                                    location=location,
                                    term=term,
                                    page=page,
                                )
                                continue
                            yield RawRecord(source_id=source_id, payload=job)

                        page += 1
