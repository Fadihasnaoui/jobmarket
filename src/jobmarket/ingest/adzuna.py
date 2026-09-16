"""Adzuna job board ingest source."""

from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from jobmarket.config import Settings, get_settings
from jobmarket.ingest.base import AbstractSource, RawRecord
from jobmarket.ingest.constants import SEARCH_TERMS
from jobmarket.ingest.http import ResilientHttpClient

logger = structlog.get_logger(__name__)

BASE_URL = "https://api.adzuna.com/v1/api/jobs"

COUNTRIES = ("fr", "de", "nl", "gb", "es", "it")

MAX_PAGE = 20
RESULTS_PER_PAGE = 50


class AdzunaSource(AbstractSource):
    """Fetch job listings from the Adzuna API."""

    name = "adzuna"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def fetch(self) -> AsyncIterator[RawRecord]:
        if not self._settings.adzuna_app_id or not self._settings.adzuna_app_key:
            raise ValueError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set")

        async with ResilientHttpClient(self.name) as client:
            for country in COUNTRIES:
                for term in SEARCH_TERMS:
                    async for record in self._fetch_search(client, country, term):
                        yield record

    async def _fetch_search(
        self,
        client: ResilientHttpClient,
        country: str,
        term: str,
    ) -> AsyncIterator[RawRecord]:
        for page in range(1, MAX_PAGE + 1):
            url = f"{BASE_URL}/{country}/search/{page}"
            params = {
                "app_id": self._settings.adzuna_app_id,
                "app_key": self._settings.adzuna_app_key,
                "results_per_page": RESULTS_PER_PAGE,
                "what": term,
                "content-type": "application/json",
            }

            response = await client.get(url, params=params)
            if response is None:
                logger.warning(
                    "adzuna_page_skipped",
                    country=country,
                    term=term,
                    page=page,
                )
                continue

            try:
                data = response.json()
            except ValueError:
                logger.warning(
                    "adzuna_invalid_json",
                    country=country,
                    term=term,
                    page=page,
                )
                continue

            results = data.get("results", [])
            if not results:
                break

            for result in results:
                job_id = result.get("id")
                if job_id is None:
                    continue
                yield RawRecord(source_id=str(job_id), payload=result)
