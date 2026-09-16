"""HTTP ingest tests with respx mocks (no network access)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from jobmarket.config import Settings
from jobmarket.ingest import http as ingest_http
from jobmarket.ingest.adzuna import AdzunaSource
from jobmarket.ingest.jooble import JoobleSource, derive_jooble_source_id
from jobmarket.ingest.remoteok import RemoteOKSource

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def fast_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_http, "MIN_DELAY_SECONDS", 0)
    monkeypatch.setattr(ingest_http, "MAX_DELAY_SECONDS", 0)


@respx.mock
async def test_remoteok_skips_legal_banner() -> None:
    payload = json.loads((FIXTURES_DIR / "remoteok_response.json").read_text(encoding="utf-8"))
    respx.get("https://remoteok.com/api").mock(
        return_value=httpx.Response(200, json=payload)
    )

    source = RemoteOKSource()
    records = [record async for record in source.fetch()]

    assert len(records) == 1
    assert records[0].source_id == "900001"


@respx.mock
async def test_adzuna_fetch_single_page(adzuna_payload: dict[str, object]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/fr/search/1" in str(request.url):
            return httpx.Response(200, json={"results": [adzuna_payload]})
        return httpx.Response(200, json={"results": []})

    route = respx.get(url__regex=r"https://api\.adzuna\.com/v1/api/jobs/.+/search/\d+").mock(
        side_effect=handler
    )

    settings = Settings(adzuna_app_id="test-id", adzuna_app_key="test-key")
    source = AdzunaSource(settings=settings)
    records = [record async for record in source.fetch()]

    assert route.call_count >= 1
    assert any(record.source_id == "4829103746" for record in records)


@respx.mock
async def test_jooble_fetch_uses_fixture_payload() -> None:
    payload = json.loads((FIXTURES_DIR / "jooble_response.json").read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        if (
            body.get("keywords") == "data scientist"
            and body.get("location") == "France"
            and body.get("page") == "1"
        ):
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"totalCount": 0, "jobs": []})

    respx.post(url__regex=r"https://jooble\.org/api/.+").mock(side_effect=handler)

    settings = Settings(jooble_api_key="test-key")
    source = JoobleSource(settings=settings)
    records = [record async for record in source.fetch()]

    assert any(record.source_id == "jooble-9001" for record in records)


def test_jooble_source_id_falls_back_to_link_hash() -> None:
    payload: dict[str, object] = {
        "link": "https://example.com/jobs/derived-id",
        "title": "Analyst",
    }
    source_id = derive_jooble_source_id(payload)
    assert len(source_id) == 64
    assert source_id == derive_jooble_source_id(payload)


def test_jooble_source_id_requires_identity() -> None:
    with pytest.raises(ValueError, match="missing both id and link"):
        derive_jooble_source_id({"title": "No link or id"})
