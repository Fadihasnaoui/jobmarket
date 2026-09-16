"""Resilient async HTTP client with rate limiting and retries."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3
MIN_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 2.5


class ResilientHttpClient:
    """HTTP client with per-source concurrency limits and host-level throttling."""

    def __init__(
        self,
        source: str,
        *,
        max_concurrent: int = 5,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.source = source
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._host_last_request: dict[str, float] = {}
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers or {})

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> ResilientHttpClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def get(self, url: str, **kwargs: Any) -> httpx.Response | None:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response | None:
        return await self._request("POST", url, **kwargs)

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response | None:
        host = urlparse(url).netloc

        async with self._semaphore:
            await self._throttle(host)
            return await self._request_with_retries(method, url, **kwargs)

    async def _throttle(self, host: str) -> None:
        now = time.monotonic()
        last = self._host_last_request.get(host)
        if last is not None:
            elapsed = now - last
            delay = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
            if elapsed < delay:
                await asyncio.sleep(delay - elapsed)
        self._host_last_request[host] = time.monotonic()

    async def _request_with_retries(
        self, method: str, url: str, **kwargs: Any
    ) -> httpx.Response | None:
        for attempt in range(1, MAX_RETRIES + 1):
            started = time.perf_counter()
            try:
                response = await self._client.request(method, url, **kwargs)
                latency_ms = round((time.perf_counter() - started) * 1000, 2)

                logger.info(
                    "http_request",
                    source=self.source,
                    method=method,
                    url=url,
                    status=response.status_code,
                    latency_ms=latency_ms,
                    attempt=attempt,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        await self._backoff(response, attempt)
                        continue
                    logger.warning(
                        "http_request_exhausted_retries",
                        source=self.source,
                        url=url,
                        status=response.status_code,
                    )
                    return None

                if response.status_code >= 400:
                    logger.warning(
                        "http_request_client_error",
                        source=self.source,
                        url=url,
                        status=response.status_code,
                    )
                    return None

                return response

            except httpx.RequestError as exc:
                latency_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.warning(
                    "http_request_error",
                    source=self.source,
                    method=method,
                    url=url,
                    latency_ms=latency_ms,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2**attempt)
                    continue
                return None

        return None

    async def _backoff(self, response: httpx.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                await asyncio.sleep(float(retry_after))
                return
            except ValueError:
                pass
        await asyncio.sleep(2**attempt)
