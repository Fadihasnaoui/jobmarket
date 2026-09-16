"""Stage A: deterministic alias matching over job description text. No LLM, no network.

Word-boundary aware and accent-insensitive. Short aliases get extra false-positive guards
because French and English text are full of short acronym-shaped fragments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from jobmarket.skills.ontology import (
    SkillsOntology,
    get_ontology,
    ontology_version,
    strip_accents,
)

MATCHER_VERSION = "deterministic-alias-matcher-v3"

_SHORT_ALIAS_MAX_LEN = 3
_SHORT_ALIAS_ALLOWED_NEIGHBORS = frozenset(" \t\n\r,;/|()[]:.")
_CONTEXT_WINDOW = 80


@dataclass(frozen=True)
class MatcherVersions:
    """Version identifiers to stamp onto enrichment runs."""

    matcher_version: str
    ontology_version: str


def get_matcher_versions(ontology: SkillsOntology | None = None) -> MatcherVersions:
    """Return explicit matcher and semantic ontology versions together."""
    return MatcherVersions(
        matcher_version=MATCHER_VERSION,
        ontology_version=ontology_version(ontology),
    )


@dataclass(frozen=True)
class SkillMatch:
    """One matched skill, with the span (in the ORIGINAL text) it was found at."""

    canonical: str
    alias: str
    start: int
    end: int


@dataclass(frozen=True)
class AliasPolicy:
    """Centralized risk policy for ambiguous aliases.

    Standard aliases use boundary matching only. Ambiguous aliases can additionally require
    uppercase evidence, nearby technical context, or blocked surrounding phrases.
    """

    requires_uppercase: bool = False
    context_terms: tuple[str, ...] = ()
    blocked_following: tuple[re.Pattern[str], ...] = ()
    blocked_preceding: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class _CompiledAlias:
    canonical: str
    alias: str
    pattern: re.Pattern[str]
    is_short: bool
    requires_case_check: bool
    policy: AliasPolicy


_AI_CONTEXT_TERMS = (
    "ai",
    "agent",
    "artificielle",
    "artificial",
    "architecture",
    "architectures",
    "data",
    "deep learning",
    "genai",
    "g?n?rative",
    "generative",
    "ia",
    "engineer",
    "intelligence",
    "learning",
    "machine learning",
    "model",
    "mod?le",
    "ml",
)
_ML_CONTEXT_TERMS = (
    "data",
    "deep learning",
    "engineering",
    "learning",
    "machine",
    "model",
    "mod?le",
    "scientist",
)
_BI_CONTEXT_TERMS = (
    "analytics",
    "business",
    "data",
    "decisionnel",
    "d?cisionnel",
    "power bi",
    "reporting",
    "tableau",
)
_CLOUD_CONTEXT_TERMS = ("cloud", "data", "platform", "vertex")
_SAP_CONTEXT_TERMS = ("bw", "consultant", "erp", "s/4", "sap", "solution")
_RAG_CONTEXT_TERMS = ("agent", "ai", "genai", "ia", "llm", "retrieval")
_LLM_CONTEXT_TERMS = ("ai", "genai", "ia", "language", "langage", "model", "mod?le", "rag")
_R_CONTEXT_TERMS = (
    "programming",
    "programmation",
    "langage",
    "language",
    "python",
    "sql",
    "rstudio",
    "statistical",
    "statistics",
    "skills",
    "competences",
    "comp?tences",
    "technologies",
)
_C_CONTEXT_TERMS = (
    "c++",
    "c/c++",
    "developpons",
    "d?veloppons",
    "embedded",
    "langage",
    "linux",
    "python",
    "rust",
    "software",
)

_ALIAS_POLICIES: dict[tuple[str, str], AliasPolicy] = {
    ("Artificial Intelligence", "ai"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_AI_CONTEXT_TERMS,
    ),
    ("Artificial Intelligence", "ia"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_AI_CONTEXT_TERMS,
    ),
    ("Artificial Intelligence", "ki"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_AI_CONTEXT_TERMS,
    ),
    ("Chef", "chef"): AliasPolicy(
        blocked_following=(
            re.compile(r"^\s+d[eu']\b", re.IGNORECASE),
            re.compile(r"^\s+d[?']\s*equipe\b", re.IGNORECASE),
            re.compile(r"^\s+d[?']\s*?quipe\b", re.IGNORECASE),
        ),
        blocked_preceding=(
            re.compile(r"\bdu\s+$", re.IGNORECASE),
            re.compile(r"\bdes\s+$", re.IGNORECASE),
        ),
    ),
    ("Machine Learning", "ml"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_ML_CONTEXT_TERMS,
    ),
    ("Business Intelligence", "bi"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_BI_CONTEXT_TERMS,
    ),
    ("R", "r"): AliasPolicy(requires_uppercase=True, context_terms=_R_CONTEXT_TERMS),
    ("C", "c"): AliasPolicy(requires_uppercase=True, context_terms=_C_CONTEXT_TERMS),
    ("SAP", "sap"): AliasPolicy(requires_uppercase=True, context_terms=_SAP_CONTEXT_TERMS),
    ("GCP", "gcp"): AliasPolicy(requires_uppercase=True, context_terms=_CLOUD_CONTEXT_TERMS),
    ("Retrieval-Augmented Generation", "rag"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_RAG_CONTEXT_TERMS,
    ),
    ("Large Language Models", "llm"): AliasPolicy(
        requires_uppercase=True,
        context_terms=_LLM_CONTEXT_TERMS,
    ),
}


def _alias_policy(canonical: str, alias: str) -> AliasPolicy:
    return _ALIAS_POLICIES.get((canonical, strip_accents(alias.casefold())), AliasPolicy())


def fold_text(text: str) -> str:
    """Casefold + strip accents for matching."""
    return strip_accents(text.casefold())


def _fold_text_with_index_map(text: str) -> tuple[str, list[int]]:
    """Fold text while mapping each folded character back to its original offset."""
    folded_chars: list[str] = []
    original_offsets: list[int] = []
    for index, char in enumerate(text):
        folded_char = strip_accents(char.casefold())
        for output_char in folded_char:
            folded_chars.append(output_char)
            original_offsets.append(index)
    return "".join(folded_chars), original_offsets


def _neighbor_ok(text: str, index: int) -> bool:
    """True if `index` is out of bounds or holds an allowed delimiter."""
    if index < 0 or index >= len(text):
        return True
    return text[index] in _SHORT_ALIAS_ALLOWED_NEIGHBORS


@lru_cache
def _compiled_aliases() -> tuple[_CompiledAlias, ...]:
    """All default ontology aliases, compiled once."""
    return _compile_ontology(get_ontology())


def _compile_ontology(ontology: SkillsOntology) -> tuple[_CompiledAlias, ...]:
    compiled: list[_CompiledAlias] = []
    for entry in ontology.entries:
        for name in (entry.canonical, *entry.aliases):
            folded = strip_accents(name.casefold())
            parts = folded.split()
            if not parts:
                continue
            is_short = len(parts) == 1 and len(parts[0]) <= _SHORT_ALIAS_MAX_LEN
            requires_case_check = is_short and parts[0].isalpha()
            body = r"\s+".join(re.escape(part) for part in parts)
            pattern = re.compile(rf"(?<!\w){body}(?!\w)", re.UNICODE)
            compiled.append(
                _CompiledAlias(
                    entry.canonical,
                    name,
                    pattern,
                    is_short,
                    requires_case_check,
                    _alias_policy(entry.canonical, name),
                )
            )
    return tuple(compiled)


def _policy_allows(
    alias_entry: _CompiledAlias,
    original_text: str,
    folded_text: str,
    original_start: int,
    original_end: int,
    folded_start: int,
    folded_end: int,
) -> bool:
    policy = alias_entry.policy
    evidence = original_text[original_start:original_end]
    if policy.requires_uppercase and evidence == evidence.lower():
        return False

    before = folded_text[max(0, folded_start - _CONTEXT_WINDOW) : folded_start]
    after = folded_text[folded_end : min(len(folded_text), folded_end + _CONTEXT_WINDOW)]
    if any(pattern.search(after) for pattern in policy.blocked_following):
        return False
    if any(pattern.search(before) for pattern in policy.blocked_preceding):
        return False
    has_context = any(term in before or term in after for term in policy.context_terms)
    return not policy.context_terms or has_context


def find_skills(text: str, ontology: SkillsOntology | None = None) -> list[SkillMatch]:
    """Find every canonical skill mentioned in `text`. One match per canonical skill."""
    if not text:
        return []

    compiled = _compiled_aliases() if ontology is None else _compile_ontology(ontology)
    folded, original_offsets = _fold_text_with_index_map(text)

    found: dict[str, SkillMatch] = {}
    for alias_entry in compiled:
        if alias_entry.canonical in found:
            continue
        for m in alias_entry.pattern.finditer(folded):
            original_start = original_offsets[m.start()]
            original_end = original_offsets[m.end() - 1] + 1
            if alias_entry.is_short:
                if not (_neighbor_ok(folded, m.start() - 1) and _neighbor_ok(folded, m.end())):
                    continue
                if alias_entry.requires_case_check and text[original_start:original_end].islower():
                    continue
            if not _policy_allows(
                alias_entry,
                text,
                folded,
                original_start,
                original_end,
                m.start(),
                m.end(),
            ):
                continue
            found[alias_entry.canonical] = SkillMatch(
                canonical=alias_entry.canonical,
                alias=alias_entry.alias,
                start=original_start,
                end=original_end,
            )
            break

    return sorted(found.values(), key=lambda sm: sm.start)


def match_skill_canonicals(text: str, ontology: SkillsOntology | None = None) -> set[str]:
    """Convenience: just the set of canonical skill names found in `text`."""
    return {m.canonical for m in find_skills(text, ontology)}
