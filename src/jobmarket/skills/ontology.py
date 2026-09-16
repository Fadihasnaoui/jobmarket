"""Loads and validates the hand-curated skills ontology (data/skills_ontology.yaml)."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_ONTOLOGY_PATH = Path(__file__).resolve().parents[3] / "data" / "skills_ontology.yaml"

CATEGORIES = frozenset(
    {
        "language",
        "framework",
        "tool",
        "platform",
        "database",
        "cloud",
        "methodology",
        "soft_skill",
        "domain",
    }
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Remove combining accent marks (cafÃƒÂ© -> cafe), preserving per-character alignment."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_alias(text: str) -> str:
    """Case-fold, strip accents, and collapse whitespace for alias matching."""
    folded = strip_accents(text.strip().casefold())
    return _WHITESPACE_PATTERN.sub(" ", folded)


class SkillEntry(BaseModel):
    """One canonical skill entry as declared in the ontology YAML."""

    model_config = ConfigDict(frozen=True)

    canonical: str
    category: str
    aliases: list[str] = []
    parent: str | None = None

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        if value not in CATEGORIES:
            raise ValueError(f"Unknown category {value!r}; must be one of {sorted(CATEGORIES)}")
        return value


class SkillsOntology:
    """Validated skills ontology with an alias -> canonical lookup."""

    def __init__(self, entries: list[SkillEntry]) -> None:
        self.entries = entries
        self._by_canonical = {entry.canonical: entry for entry in entries}
        if len(self._by_canonical) != len(entries):
            _raise_duplicate_canonical(entries)
        self._alias_to_canonical = _build_alias_index(entries)
        _validate_parents(entries, self._by_canonical)

    def resolve(self, raw_skill: str) -> str | None:
        """Return the canonical skill name for a raw skill string, or None if unmapped."""
        return self._alias_to_canonical.get(normalize_alias(raw_skill))

    def get(self, canonical: str) -> SkillEntry | None:
        return self._by_canonical.get(canonical)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, canonical: str) -> bool:
        return canonical in self._by_canonical

    def semantic_payload(self) -> list[dict[str, object]]:
        """Return ontology content in a deterministic, formatting-independent shape."""
        payload: list[dict[str, object]] = [
            {
                "canonical": entry.canonical,
                "category": entry.category,
                "aliases": sorted(entry.aliases),
                "parent": entry.parent,
            }
            for entry in self.entries
        ]
        return sorted(payload, key=lambda item: str(item["canonical"]))

    def version(self) -> str:
        """SHA-256 hash of semantic ontology content, independent of YAML formatting."""
        canonical_json = json.dumps(
            self.semantic_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _raise_duplicate_canonical(entries: list[SkillEntry]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.canonical in seen:
            raise ValueError(f"Duplicate canonical skill: {entry.canonical!r}")
        seen.add(entry.canonical)


def _build_alias_index(entries: list[SkillEntry]) -> dict[str, str]:
    index: dict[str, str] = {}
    for entry in entries:
        _claim(index, normalize_alias(entry.canonical), entry.canonical)
        for alias in entry.aliases:
            _claim(index, normalize_alias(alias), entry.canonical)
    return index


def _claim(index: dict[str, str], key: str, canonical: str) -> None:
    existing = index.get(key)
    if existing is not None and existing != canonical:
        raise ValueError(f"Alias {key!r} is claimed by both {existing!r} and {canonical!r}")
    index[key] = canonical


def _validate_parents(entries: list[SkillEntry], by_canonical: dict[str, SkillEntry]) -> None:
    for entry in entries:
        if entry.parent is not None and entry.parent not in by_canonical:
            raise ValueError(f"{entry.canonical!r} has unknown parent {entry.parent!r}")


def load_ontology(path: Path | None = None) -> SkillsOntology:
    """Load and validate the skills ontology YAML, failing loudly on any inconsistency."""
    ontology_path = path or DEFAULT_ONTOLOGY_PATH
    raw = yaml.safe_load(ontology_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Expected a YAML list at the top level of {ontology_path}")

    entries = [SkillEntry.model_validate(item) for item in raw]
    return SkillsOntology(entries)


@lru_cache
def get_ontology() -> SkillsOntology:
    """Process-wide cached ontology singleton, loaded from the default YAML path."""
    return load_ontology()


def resolve(raw_skill: str) -> str | None:
    """Resolve a raw skill string against the default ontology singleton."""
    return get_ontology().resolve(raw_skill)


def ontology_version(ontology: SkillsOntology | None = None) -> str:
    """Return the semantic SHA-256 version for an ontology."""
    return (ontology or get_ontology()).version()
