"""Parser name to class mapping."""

from __future__ import annotations

from jobmarket.parse.adzuna import AdzunaParser
from jobmarket.parse.base import AbstractParser
from jobmarket.parse.jooble import JoobleParser
from jobmarket.parse.remoteok import RemoteOKParser

PARSER_REGISTRY: dict[str, type[AbstractParser]] = {
    "adzuna": AdzunaParser,
    "remoteok": RemoteOKParser,
    "jooble": JoobleParser,
}


def get_parser(name: str) -> AbstractParser:
    """Instantiate a registered parser by source name."""
    cls = PARSER_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(PARSER_REGISTRY))
        raise ValueError(f"Unknown parser {name!r}. Known parsers: {known}")
    return cls()


def all_parser_names() -> list[str]:
    """Return all registered parser names."""
    return sorted(PARSER_REGISTRY)
