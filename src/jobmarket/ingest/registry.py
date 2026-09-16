"""Source name to ingest class mapping."""

from __future__ import annotations

from jobmarket.ingest.adzuna import AdzunaSource
from jobmarket.ingest.base import AbstractSource
from jobmarket.ingest.jooble import JoobleSource
from jobmarket.ingest.remoteok import RemoteOKSource

SOURCE_REGISTRY: dict[str, type[AbstractSource]] = {
    "adzuna": AdzunaSource,
    "remoteok": RemoteOKSource,
    "jooble": JoobleSource,
}


def get_source(name: str) -> AbstractSource:
    """Instantiate a registered ingest source by name."""
    cls = SOURCE_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(SOURCE_REGISTRY))
        raise ValueError(f"Unknown source {name!r}. Known sources: {known}")
    return cls()


def all_source_names() -> list[str]:
    """Return all registered source names."""
    return sorted(SOURCE_REGISTRY)
