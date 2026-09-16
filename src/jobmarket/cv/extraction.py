"""Deterministic CV skill extraction using the existing matcher."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from jobmarket.cv.attributes import extract_candidate_attributes
from jobmarket.cv.profile import (
    CvProfile,
    CvSkill,
    EvidenceSpan,
    ExtractionVersions,
    ParsedDocument,
    RoleInference,
)
from jobmarket.cv.quality import analyze_cv_text, section_for_offset
from jobmarket.db.models import Skill
from jobmarket.skills.matcher import find_skills, get_matcher_versions
from jobmarket.skills.ontology import SkillsOntology, load_ontology, normalize_alias

CV_EXTRACTION_VERSION = "cv-deterministic-extractor-v1"


def extract_cv_profile(
    document: ParsedDocument,
    *,
    ontology: SkillsOntology | None = None,
    skill_ids: dict[str, int] | None = None,
) -> CvProfile:
    """Build a deterministic CV profile with grounded ontology skills only."""
    actual_ontology = ontology or load_ontology()
    actual_skill_ids = skill_ids or _default_skill_ids(actual_ontology)
    versions = get_matcher_versions(actual_ontology)
    prelim_matches = find_skills(document.text, actual_ontology)
    prelim_aliases = {match.alias for match in prelim_matches}
    analysis = analyze_cv_text(
        document.text,
        {match.canonical for match in prelim_matches},
        prelim_aliases,
        document.blocks,
    )
    skills: list[CvSkill] = []
    rejected_ambiguous: list[str] = _isolated_ambiguous_aliases(document.text)
    for match in prelim_matches:
        section = section_for_offset(analysis.sections, match.start)
        if match.canonical in {"R", "C"} and not _ambiguous_language_allowed(
            document.text, match.start, match.end, section
        ):
            rejected_ambiguous.append(match.alias)
            continue
        evidence_text = document.text[match.start : match.end]
        page, page_start, page_end = _page_offsets(document, match.start, match.end)
        skill_id = actual_skill_ids.get(match.canonical)
        if skill_id is None:
            raise ValueError(f"No ontology skill ID for {match.canonical!r}")
        confidence = 1.0 if section == "skills" else 0.82 if section else 0.7
        skills.append(
            CvSkill(
                canonical_skill=match.canonical,
                ontology_skill_id=skill_id,
                matched_alias=match.alias,
                extraction_method="deterministic",
                evidence=EvidenceSpan(
                    evidence_text=evidence_text,
                    page=page,
                    document_start=match.start,
                    document_end=match.end,
                    page_start=page_start,
                    page_end=page_end,
                ),
                matcher_version=versions.matcher_version,
                ontology_version=versions.ontology_version,
                validation_status="deterministic",
                confidence=confidence,
            )
        )
    skills = sorted(
        skills,
        key=lambda skill: (skill.evidence.document_start, skill.canonical_skill),
    )
    skill_names = {skill.canonical_skill for skill in skills}
    business_skills = analysis.business_skills
    candidate = extract_candidate_attributes(
        document.text,
        skill_names | business_skills,
        actual_ontology,
    )
    if (
        analysis.education is not None
        and analysis.education.education_status == "graduated"
    ):
        candidate = candidate.model_copy(
            update={"education_status": analysis.education.education_status}
        )
    if analysis.experiences:
        internship_count = sum(
            1 for item in analysis.experiences if item.entry_type == "internship"
        )
        professional_count = sum(1 for item in analysis.experiences if item.entry_type == "job")
        months = [item.duration_months for item in analysis.experiences if item.duration_months]
        experience = candidate.experience.model_copy(
            update={
                "internship_count": max(candidate.experience.internship_count, internship_count),
                "professional_experience_count": max(
                    candidate.experience.professional_experience_count, professional_count
                ),
                "total_months": sum(months) if months else candidate.experience.total_months,
                "total_years": round(sum(months) / 12, 2)
                if months
                else candidate.experience.total_years,
                "evidence": sorted({item.evidence for item in analysis.experiences}),
            }
        )
        candidate = candidate.model_copy(update={"experience": experience})
    if analysis.domains:
        domains = {
            *candidate.preferred_domains,
            *_promotable_domains(analysis.domains, document.text),
        }
        technical_domains = {"AI", "Data Science", "Machine Learning", "Software Engineering"}
        technical_skills = {
            "Python",
            "Data Science",
            "Machine Learning",
            "Artificial Intelligence",
        }
        if not skill_names & technical_skills:
            domains -= technical_domains
        candidate = candidate.model_copy(update={"preferred_domains": sorted(domains)})
    if analysis.education is not None and analysis.education.graduated and analysis.experiences or (
        candidate.career_level == "unknown"
        and analysis.education is not None
        and analysis.education.education_level is not None
    ):
        candidate = candidate.model_copy(update={"career_level": "junior"})
    quality = analysis.quality.model_copy(
        update={"rejected_ambiguous_aliases": sorted(set(rejected_ambiguous))}
    )
    technical_profile_skills = {
        "Python",
        "Data Science",
        "Machine Learning",
        "Artificial Intelligence",
    }
    roles = analysis.roles
    if not skill_names & technical_profile_skills:
        roles = [role for role in roles if role.value != "Data / AI"]
    return CvProfile(
        document=document,
        versions=ExtractionVersions(
            parser_version=document.parser_version,
            matcher_version=versions.matcher_version,
            ontology_version=versions.ontology_version,
        ),
        skills=skills,
        candidate=candidate,
        warnings=document.warnings,
        sections=analysis.sections,
        education_detail=analysis.education,
        experience_entries=analysis.experiences,
        inferred_roles=roles,
        inferred_domains=analysis.domains,
        extraction_quality=quality,
        extraction_traces=analysis.traces,
    )

def _promotable_domains(inferences: list[RoleInference], text: str) -> set[str]:
    """Return inferred domains supported by explicit, non-ambiguous evidence."""
    return {
        inference.value
        for inference in inferences
        if _domain_inference_is_promotable(inference, text)
    }


def _domain_inference_is_promotable(inference: RoleInference, text: str) -> bool:
    canonical = normalize_alias(inference.value)
    evidence = {
        normalized
        for raw_evidence in inference.evidence
        if (normalized := normalize_alias(raw_evidence))
    }
    aliases = evidence - {canonical}
    if aliases:
        return True
    if len(canonical.split()) == 1:
        return canonical in evidence
    return canonical in evidence and not _looks_like_isolated_experience_title(text, canonical)


def _looks_like_isolated_experience_title(text: str, canonical: str) -> bool:
    for line in text.splitlines():
        normalized_line = normalize_alias(line)
        if canonical not in normalized_line:
            continue
        if len(line) > 220:
            continue
        has_month = re.search(
            r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
            line,
            re.IGNORECASE,
        )
        if " | " in line or has_month:
            return True
        if re.search(r"\b\d{4}\b", line) and re.search(r"\s[-–—]\s", line):
            return True
    return False


async def load_skill_ids(
    session: AsyncSession,
    ontology: SkillsOntology | None = None,
) -> dict[str, int]:
    """Load canonical skill IDs once for CV extraction and matching."""
    actual_ontology = ontology or load_ontology()
    expected = {entry.canonical for entry in actual_ontology.entries}
    rows = await session.execute(
        select(Skill.canonical, Skill.id).where(Skill.canonical.in_(expected))
    )
    mapping = {canonical: int(skill_id) for canonical, skill_id in rows.all()}
    missing = sorted(expected - set(mapping))
    if missing:
        raise ValueError(f"Skills table is missing ontology canonicals: {', '.join(missing[:10])}")
    return mapping


def _isolated_ambiguous_aliases(text: str) -> list[str]:
    import re

    aliases: list[str] = []
    for match in re.finditer(r"(?m)^\s*([RC])\s*$|(?<!\w)([RC])(?!\w)", text):
        value = match.group(1) or match.group(2)
        if value and not _ambiguous_language_allowed(text, match.start(), match.end(), None):
            aliases.append(value)
    return sorted(set(aliases))


def _ambiguous_language_allowed(
    text: str,
    start: int,
    end: int,
    section: str | None,
) -> bool:
    if section == "skills":
        return True
    window = text[max(0, start - 60) : min(len(text), end + 60)].casefold()
    patterns = (
        "r programming",
        "programming language r",
        "langage r",
        "python / r / sql",
        "python, r, sql",
        "r/python",
        "r / python",
        "rstudio",
        "statistical analysis using r",
        "c programming",
        "programming language c",
        "langage c",
        "c/c++",
        "c, c++",
    )
    return any(pattern in window for pattern in patterns)


def _page_offsets(
    document: ParsedDocument,
    start: int,
    end: int,
) -> tuple[int | None, int | None, int | None]:
    for page in document.pages:
        if start >= page.start_offset and end <= page.end_offset:
            if page.page is None:
                return None, None, None
            page_start = start - page.start_offset
            page_end = end - page.start_offset
            if page.text[page_start:page_end] != document.text[start:end]:
                raise ValueError("page-relative evidence slice does not match document evidence")
            return page.page, page_start, page_end
    return None, None, None


def _default_skill_ids(ontology: SkillsOntology) -> dict[str, int]:
    return {entry.canonical: index for index, entry in enumerate(ontology.entries, start=1)}


__all__ = ["CV_EXTRACTION_VERSION", "extract_cv_profile", "load_skill_ids"]
