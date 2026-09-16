"""Deterministic embedding text builders for jobs and CV profiles."""

from __future__ import annotations

import re
from collections.abc import Iterable
from html import unescape
from typing import Any

from jobmarket.cv.attributes import extract_job_attributes
from jobmarket.cv.profile import CvProfile, EducationDetail, ExperienceEntry

_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_DESCRIPTION_LIMIT = 1500


def build_job_embedding_text(job: object, enrichment: object) -> str:
    """Compose deterministic semantic text for one normalized job.

    The text is intentionally structured to align with CV embedding text while preserving
    lexical evidence as the authoritative skill source.
    """
    title = _clean(_get(job, "title"))
    description = _clean(_get(job, "description"))[:_DESCRIPTION_LIMIT]
    skills = _sorted_unique(_iter_strings(_get(enrichment, "skills", [])))
    attrs = extract_job_attributes(
        title=title,
        description=description,
        contract_type=_optional_clean(_get(job, "contract_type")),
        is_remote=_get(job, "is_remote"),
        country=_optional_clean(_get(job, "country")),
        city=_optional_clean(_get(job, "city")),
        skill_canonicals=set(skills),
    )
    lines: list[str] = []
    _append(lines, "Title", title)
    if attrs.career_level != "unknown":
        _append(lines, "Career level", attrs.career_level)
    _append(lines, "Skills", ", ".join(skills))
    _append(lines, "Domain", ", ".join(attrs.domains))
    _append(lines, "Contract", _optional_clean(_get(job, "contract_type")))
    if attrs.remote_mode is not None:
        _append(lines, "Work mode", attrs.remote_mode)
    elif _get(job, "is_remote") is True:
        _append(lines, "Work mode", "remote")
    elif _get(job, "is_remote") is False:
        _append(lines, "Work mode", "on_site")
    if title and description.casefold().startswith(title.casefold()):
        description = description[len(title) :].strip(" -:|\n\t")
    _append(lines, "Description", description)
    return "\n".join(lines)


def build_cv_embedding_text(profile: CvProfile) -> str:
    """Compose deterministic in-memory semantic text for a parsed CV profile.

    Includes the actual grounded CV prose (experience/education evidence quotes
    from `experience_entries`/`education_detail`), not just canonical skill/domain
    tags — `profile.education`/`roles`/`experience_statements` are legacy fields no
    extractor ever populates, so relying on them left semantic similarity comparing
    normalized labels only, never the CV's real text.
    """
    candidate = profile.candidate
    skills = _sorted_unique(skill.canonical_skill for skill in profile.skills)
    domains = _sorted_unique(candidate.preferred_domains)
    languages = _sorted_unique(candidate.programming_languages)
    databases = _sorted_unique(candidate.databases)
    education = _education_lines(profile.education_detail)
    experience = _experience_lines(profile.experience_entries)
    lines: list[str] = []
    if candidate.career_level != "unknown":
        _append(lines, "Career level", candidate.career_level)
    _append(lines, "Skills", ", ".join(skills))
    _append(lines, "Domain", ", ".join(domains))
    _append(lines, "Programming languages", ", ".join(languages))
    _append(lines, "Databases", ", ".join(databases))
    _append(lines, "Education", "; ".join(education))
    _append(lines, "Experience", "; ".join(experience))
    return "\n".join(lines)


def _education_lines(detail: EducationDetail | None) -> list[str]:
    if detail is None:
        return []
    header = ", ".join(
        part
        for part in (
            _optional_clean(detail.education_level),
            _optional_clean(detail.education_field),
            _optional_clean(detail.institution),
            _optional_clean(detail.education_status),
        )
        if part
    )
    quotes = "; ".join(_sorted_unique(detail.evidence))
    return [part for part in (header, quotes) if part]


def _experience_lines(entries: list[ExperienceEntry]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        employer = _optional_clean(entry.employer)
        domain = _optional_clean(entry.domain)
        header = " ".join(
            part
            for part in (
                _optional_clean(entry.title),
                f"at {employer}" if employer else None,
                f"({domain})" if domain else None,
            )
            if part
        )
        evidence = _optional_clean(entry.evidence)
        line = ": ".join(part for part in (header, evidence) if part)
        if line:
            lines.append(line)
    return lines


def _append(lines: list[str], label: str, value: str | None) -> None:
    cleaned = _optional_clean(value)
    if cleaned:
        lines.append(f"{label}: {cleaned}")


def _get(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _iter_strings(values: object) -> Iterable[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, Iterable):
        return [str(value) for value in values if value is not None]
    return [str(values)]


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({_clean(value) for value in values if _clean(value)}, key=str.casefold)


def _optional_clean(value: object) -> str | None:
    if value is None:
        return None
    cleaned = _clean(str(value))
    return cleaned or None


def _clean(value: object) -> str:
    text = unescape(str(value or ""))
    text = _HTML_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


__all__ = ["build_cv_embedding_text", "build_job_embedding_text"]
