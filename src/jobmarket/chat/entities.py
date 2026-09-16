"""Deterministic entity extraction from a raw chat question.

Every extractor here is a plain keyword/regex match against known, real values (the
skill ontology's own alias table, the model's trained country codes, the existing
`_SENIORITY_PATTERNS` from `api/stats_routes.py`, a fixed contract-type/role-title
vocabulary) — never an LLM guess. This is what lets the SQL/aggregate answer path
build a real, parameterized query instead of asking an LLM to invent a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from jobmarket.api.stats_routes import _SENIORITY_PATTERNS
from jobmarket.skills.ontology import SkillsOntology, normalize_alias

# Explicit country aliases supported by the job corpus and SQL answer path.
_COUNTRY_ALIASES: dict[str, str] = {
    "fr": "fr",
    "france": "fr",
    "french": "fr",
    "français": "fr",
    "francais": "fr",
    "de": "de",
    "germany": "de",
    "german": "de",
    "deutschland": "de",
    "allemagne": "de",
    "es": "es",
    "spain": "es",
    "spanish": "es",
    "espagne": "es",
    "españa": "es",
    "gb": "gb",
    "uk": "gb",
    "u.k.": "gb",
    "united kingdom": "gb",
    "britain": "gb",
    "british": "gb",
    "england": "gb",
    "angleterre": "gb",
    "it": "it",
    "italy": "it",
    "italian": "it",
    "italie": "it",
    "nl": "nl",
    "netherlands": "nl",
    "dutch": "nl",
    "holland": "nl",
    "pays-bas": "nl",
    "pays bas": "nl",
}

_CONTRACT_TYPE_ALIASES: dict[str, str] = {
    "full_time": "full_time",
    "full-time": "full_time",
    "full time": "full_time",
    "cdi": "full_time",
    "permanent": "full_time",
    "part_time": "part_time",
    "part-time": "part_time",
    "part time": "part_time",
    "contract": "contract",
    "contractor": "contract",
    "freelance": "contract",
    "cdd": "contract",
    "temporary": "contract",
}

# A modest, curated set of common tech role titles — matched as a case-insensitive
# substring of the question against `Job.title`. Not exhaustive by design: an
# unrecognized title falls back to no title filter (a broader, still-honest query)
# rather than a wrong guess.
_ROLE_TITLES: tuple[str, ...] = (
    "data scientist",
    "data engineer",
    "data analyst",
    "machine learning engineer",
    "ml engineer",
    "ai engineer",
    "software engineer",
    "software developer",
    "backend developer",
    "backend engineer",
    "frontend developer",
    "frontend engineer",
    "full stack developer",
    "fullstack developer",
    "devops engineer",
    "cloud engineer",
    "security engineer",
    "product manager",
    "project manager",
    "business analyst",
    "qa engineer",
    "data architect",
    "solutions architect",
)

_SKILL_WINDOW_MAX_WORDS = 4


@dataclass(frozen=True)
class ExtractedEntities:
    skill: str | None = None
    country: str | None = None
    seniority: str | None = None
    contract_type: str | None = None
    role_title: str | None = None
    skill_matches: list[str] = field(default_factory=list)


def extract_country(question: str) -> str | None:
    lowered = question.casefold()
    for alias, code in sorted(_COUNTRY_ALIASES.items(), key=lambda item: -len(item[0])):
        # Bare ISO codes are not country names in ordinary prose: in particular,
        # the English pronoun "it" must never become the Italy filter.
        if len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", lowered):
            return code
    explicit_code = re.search(
        r"\b(?:country|pays)\s*(?:=|:|is|est)\s*([a-z]{2})\b", lowered
    )
    if explicit_code is not None and explicit_code.group(1) in {"fr", "de", "es", "gb", "it", "nl"}:
        return explicit_code.group(1)
    return None


def extract_seniority(question: str) -> str | None:
    """Reuses `_SENIORITY_PATTERNS`' exact keyword groups so the SQL filter and this
    router-side detection can never drift apart -- but that dict's patterns use `\\m`/
    `\\M`, Postgres's own word-boundary escapes (correct for the `~*` SQL filter they
    were written for, see `stats_routes.py`), which Python's `re` module doesn't
    understand. Translated to Python's `\\b` here rather than duplicating the keyword
    lists with different syntax.
    """
    lowered = question.casefold()
    for level, pattern in _SENIORITY_PATTERNS.items():
        python_pattern = pattern.replace(r"\m", r"\b").replace(r"\M", r"\b")
        if re.search(python_pattern, lowered, re.IGNORECASE):
            return level
    return None


def extract_contract_type(question: str) -> str | None:
    lowered = question.casefold()
    for alias, value in sorted(_CONTRACT_TYPE_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return value
    return None


def extract_role_title(question: str) -> str | None:
    lowered = question.casefold()
    for title in sorted(_ROLE_TITLES, key=len, reverse=True):
        if title in lowered:
            return title
    return None


def extract_skills(question: str, ontology: SkillsOntology, *, limit: int = 3) -> list[str]:
    """Return canonical skill names mentioned in the question, longest phrase first.

    Tries every word window from `_SKILL_WINDOW_MAX_WORDS` words down to 1, resolving
    each through the ontology's own alias table (`ontology.resolve`, the same
    normalize+lookup every other skill-matching path in this project already uses —
    see `skills/ontology.py`). Longer windows are tried first so "machine learning"
    resolves before its substrings "machine"/"learning" would.
    """
    words = re.findall(r"[\w+.#/-]+", question, re.UNICODE)
    found: list[str] = []
    seen_spans: set[tuple[int, int]] = set()
    for window in range(_SKILL_WINDOW_MAX_WORDS, 0, -1):
        for start in range(0, len(words) - window + 1):
            span = (start, start + window)
            if any(s <= start < e or s < start + window <= e for s, e in seen_spans):
                continue
            phrase = " ".join(words[start : start + window])
            canonical = ontology.resolve(phrase)
            if canonical is not None and canonical not in found:
                found.append(canonical)
                seen_spans.add(span)
                if len(found) >= limit:
                    return found
    return found


def extract_entities(question: str, ontology: SkillsOntology) -> ExtractedEntities:
    skills = extract_skills(question, ontology)
    return ExtractedEntities(
        skill=skills[0] if skills else None,
        country=extract_country(question),
        seniority=extract_seniority(question),
        contract_type=extract_contract_type(question),
        role_title=extract_role_title(question),
        skill_matches=skills,
    )


def normalize_for_matching(text: str) -> str:
    return normalize_alias(text)


__all__ = [
    "ExtractedEntities",
    "extract_contract_type",
    "extract_country",
    "extract_entities",
    "extract_role_title",
    "extract_seniority",
    "extract_skills",
    "normalize_for_matching",
]
