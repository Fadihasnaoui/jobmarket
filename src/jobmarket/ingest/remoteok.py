"""RemoteOK job board ingest source."""

from __future__ import annotations

from collections.abc import AsyncIterator

import structlog

from jobmarket.ingest.base import AbstractSource, RawRecord
from jobmarket.ingest.http import ResilientHttpClient

logger = structlog.get_logger(__name__)

API_URL = "https://remoteok.com/api"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class RemoteOKSource(AbstractSource):
    """Fetch job listings from the RemoteOK API."""

    name = "remoteok"

    async def fetch(self) -> AsyncIterator[RawRecord]:
        headers = {"User-Agent": USER_AGENT}
        async with ResilientHttpClient(self.name, headers=headers) as client:
            response = await client.get(API_URL)
            if response is None:
                logger.warning("remoteok_fetch_failed")
                return

            try:
                data = response.json()
            except ValueError:
                logger.warning("remoteok_invalid_json")
                return

            if not isinstance(data, list):
                logger.warning("remoteok_unexpected_payload", payload_type=type(data).__name__)
                return

            for index, item in enumerate(data):
                if index == 0:
                    continue
                if not isinstance(item, dict):
                    continue

                job_id = item.get("id")
                if job_id is None:
                    continue

                yield RawRecord(source_id=str(job_id), payload=item)
