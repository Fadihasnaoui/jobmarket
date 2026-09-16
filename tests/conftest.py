"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobmarket.db.session import engine

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict[str, object]:
    path = FIXTURES_DIR / name
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Expected object fixture in {name}"
        raise TypeError(msg)
    return data


def load_json(name: str) -> object:
    path = FIXTURES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def adzuna_payload() -> dict[str, object]:
    return load_fixture("adzuna_job.json")


@pytest.fixture
def remoteok_payload() -> dict[str, object]:
    data = load_json("remoteok_response.json")
    assert isinstance(data, list)
    return data[1]


@pytest.fixture
def jooble_payload() -> dict[str, object]:
    data = load_json("jooble_response.json")
    assert isinstance(data, dict)
    jobs = data["jobs"]
    assert isinstance(jobs, list)
    job = jobs[0]
    assert isinstance(job, dict)
    return job


@pytest.fixture(autouse=True)
async def dispose_async_engine_after_test() -> None:
    """Prevent asyncpg pool from reusing connections bound to a closed event loop."""
    yield
    await engine.dispose()
