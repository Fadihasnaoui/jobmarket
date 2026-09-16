"""Deterministic candidate and job attribute extraction for recommendation scoring."""

from __future__ import annotations

import re

from jobmarket.cv.profile import (
    CandidateAttributes,
    CareerLevel,
    ExperienceSummary,
    JobAttributeProvenance,
    JobAttributes,
    JobType,
)
from jobmarket.skills.ontology import SkillsOntology, normalize_alias

_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("student", r"\b(student|etudiant|eleve ingenieur)\b"),
    ("Intern", r"\b(intern|internship|stagiaire|stage)\b"),
    ("Junior Data Scientist", r"\bjunior\s+data\s+scientist\b"),
    ("Data Scientist", r"\bdata\s+scientist\b"),
    ("Data Engineer", r"\bdata\s+engineer|ingenieur\s+data\b"),
    ("AI Engineer", r"\b(ai|ia)\s+engineer|ingenieur\s+ia\b"),
    ("Backend Developer", r"\bbackend\s+(developer|engineer)|developpeur\s+backend\b"),
)
_EDUCATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("PhD", r"\b(phd|doctorat|doctoral)\b"),
    ("Student", r"\b(student|etudiant|master|msc|bachelor|licence|ecole)\b"),
    ("Graduated", r"\b(graduated|diplome|graduate)\b"),
)
_JOB_LEVEL_PATTERNS: tuple[tuple[CareerLevel, str], ...] = (
    ("principal", r"\b(principal|staff)\b"),
    ("lead", r"\b(lead|tech lead|leader technique)\b"),
    ("manager", r"\b(manager|responsable|head of|chef de projet)\b"),
    ("senior", r"\b(senior|sr\.?|confirme|experimente)\b"),
    ("junior", r"\b(junior|debutant)\b"),
    ("internship", r"\b(stage|stagiaire|internship|intern)\b"),
    ("apprenticeship", r"\b(alternance|apprenti(?:e|s)?|apprentice)\b"),
    ("consultant", r"\bconsultant\b"),
)
_JOB_TYPE_PATTERNS: tuple[tuple[JobType, str], ...] = (
    ("freelance", r"\b(freelance|independant)\b"),
    ("stage", r"\b(stage|stagiaire|internship)\b"),
    ("apprenticeship", r"\b(alternance|apprenti(?:e|s)?|apprentice)\b"),
    ("cdi", r"\bcdi\b"),
    ("cdd", r"\bcdd\b"),
    ("hybrid", r"\b(hybrid|hybride)\b"),
    ("remote", r"\b(remote|teletravail|fully remote|full remote)\b"),
    ("on_site", r"\b(on[- ]?site|presentiel)\b"),
)
_EXPERIENCE_RANGE_PATTERN = re.compile(
    r"(?P<min>\d+(?:[.,]\d+)?)\s*(?:-|a|to)\s*(?P<max>\d+(?:[.,]\d+)?)\s*"
    r"(?:years?|ans?|annees?)",
    re.IGNORECASE,
)
_EXPERIENCE_MIN_PATTERN = re.compile(
    r"(?:(?:minimum|min|at least|au moins)\s*)?(?P<years>\d+(?:[.,]\d+)?)\s*\+?\s*"
    r"(?:years?|ans?|annees?)",
    re.IGNORECASE,
)
_MONTH_PATTERN = re.compile(r"(?P<months>\d+)\s*(?:months?|mois)", re.IGNORECASE)

_DOMAIN_SKILLS = {
    "Artificial Intelligence": "AI",
    "Data Engineering": "Data Engineering",
    "Data Science": "Data Science",
    "Machine Learning": "Machine Learning",
    "Computer Vision": "Computer Vision",
    "Natural Language Processing": "NLP",
    "Retrieval-Augmented Generation": "RAG",
    "Generative AI": "Generative AI",
    "Backend Development": "Backend",
    "Cloud Computing": "Cloud",
    "DevOps": "DevOps",
    "Cybersecurity": "Security",
}
_DOMAIN_TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Data Science", r"\bdata science|data scientist|data analyst\b"),
    ("Data Engineering", r"\bdata engineering|data engineer|ingenieur data\b"),
    ("Machine Learning", r"\bmachine learning|\bml\b"),
    ("AI", r"\bai\b|\bia\b|intelligence artificielle"),
    ("Computer Vision", r"computer vision|vision par ordinateur"),
    ("NLP", r"\bnlp\b|natural language|traitement du langage"),
    ("RAG", r"\brag\b|retrieval"),
    ("Generative AI", r"generative ai|genai|ia generative"),
    ("Software Engineering", r"developer|developpeur|full stack|informatique|software|logiciel"),
    ("Backend", r"backend|back-end"),
    ("Cloud", r"cloud|aws|azure|gcp"),
    ("DevOps", r"devops|sre|ci/cd"),
    ("Security", r"security|securite|cybersecurity|cybersecurite"),
)
_LOCATION_CITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Paris", r"\bparis\b"),
    ("Nantes", r"\bnantes\b"),
    ("Lyon", r"\blyon\b"),
    ("Marseille", r"\bmarseille\b"),
    ("Toulouse", r"\btoulouse\b"),
)
_LOCATION_COUNTRY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("fr", r"\b(france|fr)\b"),
    ("us", r"\b(united states|usa|us)\b"),
    ("gb", r"\b(united kingdom|uk|gb)\b"),
)


def extract_candidate_attributes(
    text: str,
    skill_canonicals: set[str],
    ontology: SkillsOntology,
) -> CandidateAttributes:
    """Extract deterministic, non-sensitive candidate attributes from explicit CV text."""
    normalized = normalize_alias(text)
    role = _first_pattern(text, _ROLE_PATTERNS)
    education_status = _first_pattern(text, _EDUCATION_PATTERNS)
    experience = _candidate_experience(text)
    skill_buckets = _skill_buckets(skill_canonicals, ontology)
    domains = _domains_from_skills(skill_canonicals) | _domains_from_text(normalized)
    career_level = _candidate_career_level(normalized, role, education_status, experience)
    return CandidateAttributes(
        current_role=role,
        career_level=career_level,
        education_status=education_status,
        experience=experience,
        preferred_domains=sorted(domains),
        programming_languages=skill_buckets["language"],
        databases=skill_buckets["database"],
        frameworks=skill_buckets["framework"],
        cloud_platforms=skill_buckets["cloud"],
        preferred_country=_first_pattern(text, _LOCATION_COUNTRY_PATTERNS),
        preferred_city=_first_pattern(text, _LOCATION_CITY_PATTERNS),
        remote_preference=_candidate_remote_preference(normalized),
    )


def extract_job_attributes(
    *,
    title: str,
    description: str,
    contract_type: str | None,
    is_remote: bool | None,
    country: str | None,
    city: str | None,
    skill_canonicals: set[str],
) -> JobAttributes:
    """Extract deterministic job attributes from job title, description, and metadata."""
    combined_text = f"{title}\n\n{description}\n\n{contract_type or ''}"
    normalized = normalize_alias(combined_text)
    level, career_provenance = _job_career_level_with_provenance(title, description)
    exp_min, exp_max = _job_experience_requirement(combined_text)
    job_types, job_type_provenance = _job_types_with_provenance(
        title=title,
        description=description,
        contract_type=contract_type,
        is_remote=is_remote,
    )
    remote_mode = _remote_mode(normalized, is_remote, job_types)
    domains = _domains_from_skills(skill_canonicals) | _domains_from_text(normalized)
    return JobAttributes(
        career_level=level,
        experience_min_years=exp_min,
        experience_max_years=exp_max,
        job_types=sorted(job_types),
        remote_mode=remote_mode,
        country=country,
        city=city,
        domains=sorted(domains),
        career_level_provenance=career_provenance,
        job_type_provenance=job_type_provenance,
    )


def _first_pattern(text: str, patterns: tuple[tuple[str, str], ...]) -> str | None:
    normalized = normalize_alias(text)
    for value, pattern in patterns:
        if re.search(pattern, normalized, re.IGNORECASE):
            return value
    return None


def _candidate_experience(text: str) -> ExperienceSummary:
    normalized = normalize_alias(text)
    internship_count = len(re.findall(r"\b(stage|stagiaire|internship|intern)\b", normalized))
    professional_pattern = (
        r"\b(professional experience|work experience|experience professionnelle)\b"
    )
    professional_count = len(re.findall(professional_pattern, normalized))
    months: list[int] = [int(match.group("months")) for match in _MONTH_PATTERN.finditer(text)]
    for match in _EXPERIENCE_MIN_PATTERN.finditer(text):
        months.append(round(float(match.group("years").replace(",", ".")) * 12))
    total_months = max(months) if months else None
    evidence = [match.group(0) for match in _MONTH_PATTERN.finditer(text)]
    evidence.extend(match.group(0) for match in _EXPERIENCE_MIN_PATTERN.finditer(text))
    return ExperienceSummary(
        internship_count=internship_count,
        professional_experience_count=professional_count,
        total_months=total_months,
        total_years=round(total_months / 12, 2) if total_months is not None else None,
        evidence=sorted(set(evidence)),
    )


def _candidate_career_level(
    normalized_text: str,
    role: str | None,
    education_status: str | None,
    experience: ExperienceSummary,
) -> CareerLevel:
    if re.search(r"\b(student|etudiant|eleve ingenieur)\b", normalized_text):
        return "student"
    if role == "Intern" or re.search(r"\b(intern|stagiaire)\b", normalized_text):
        return "internship"
    if re.search(r"\bjunior\b", normalized_text):
        return "junior"
    if re.search(r"\bsenior\b", normalized_text):
        return "senior"
    if education_status == "Student":
        return "student"
    years = experience.total_years
    if years is None:
        return "unknown"
    if years < 1:
        return "junior"
    if years < 4:
        return "mid"
    return "senior"


def _job_career_level_with_provenance(
    title: str,
    description: str,
) -> tuple[CareerLevel, list[JobAttributeProvenance]]:
    for source, text in (("title", title), ("description", description)):
        normalized = normalize_alias(text)
        for level, pattern in _JOB_LEVEL_PATTERNS:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match is not None:
                return level, [
                    JobAttributeProvenance(
                        value=level,
                        source=source,
                        evidence=match.group(0),
                    )
                ]
    return "unknown", []


def _job_experience_requirement(text: str) -> tuple[float | None, float | None]:
    range_match = _EXPERIENCE_RANGE_PATTERN.search(text)
    if range_match:
        return _float(range_match.group("min")), _float(range_match.group("max"))
    candidates = [
        value
        for match in _EXPERIENCE_MIN_PATTERN.finditer(text)
        if (value := _float(match.group("years"))) is not None
    ]
    if not candidates:
        return None, None
    return min(candidates), None


def _job_types_with_provenance(
    *,
    title: str,
    description: str,
    contract_type: str | None,
    is_remote: bool | None,
) -> tuple[set[JobType], list[JobAttributeProvenance]]:
    values: set[JobType] = set()
    provenance: list[JobAttributeProvenance] = []
    for source, text in (("title", title), ("description", description)):
        normalized = normalize_alias(text)
        for job_type, pattern in _JOB_TYPE_PATTERNS:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                values.add(job_type)
                provenance.append(
                    JobAttributeProvenance(
                        value=job_type,
                        source=source,
                        evidence=match.group(0),
                    )
                )
    if contract_type:
        contract = normalize_alias(contract_type)
        metadata_rules: tuple[tuple[JobType, str], ...] = (
            ("cdi", "cdi"),
            ("cdd", "cdd"),
            ("freelance", "freelance"),
            ("stage", "stage"),
            ("apprenticeship", "alternance"),
        )
        for value, token in metadata_rules:
            if token in contract:
                values.add(value)
                provenance.append(
                    JobAttributeProvenance(
                        value=value,
                        source="metadata",
                        evidence=contract_type,
                    )
                )
    if is_remote is True:
        values.add("remote")
        provenance.append(
            JobAttributeProvenance(value="remote", source="metadata", evidence="is_remote=True")
        )
    elif is_remote is False:
        values.add("on_site")
        provenance.append(
            JobAttributeProvenance(value="on_site", source="metadata", evidence="is_remote=False")
        )
    return values, _dedupe_provenance(provenance)


def _remote_mode(
    normalized_text: str,
    is_remote: bool | None,
    job_types: set[JobType],
) -> JobType | None:
    if "hybrid" in job_types:
        return "hybrid"
    if is_remote is True or "remote" in job_types:
        return "remote"
    if is_remote is False or "on_site" in job_types:
        return "on_site"
    if re.search(r"\b(teletravail|teletravail|remote)\b", normalized_text):
        return "remote"
    return None


def _candidate_remote_preference(normalized_text: str) -> bool | None:
    if re.search(r"\b(remote|teletravail|teletravail)\b", normalized_text):
        return True
    if re.search(r"\b(on[- ]?site|presentiel)\b", normalized_text):
        return False
    return None


def _domains_from_skills(skill_canonicals: set[str]) -> set[str]:
    return {domain for skill, domain in _DOMAIN_SKILLS.items() if skill in skill_canonicals}


def _domains_from_text(normalized_text: str) -> set[str]:
    return {
        domain
        for domain, pattern in _DOMAIN_TEXT_PATTERNS
        if re.search(pattern, normalized_text, re.IGNORECASE)
    }


def _dedupe_provenance(
    provenance: list[JobAttributeProvenance],
) -> list[JobAttributeProvenance]:
    seen: set[tuple[str, str, str]] = set()
    result: list[JobAttributeProvenance] = []
    for item in provenance:
        key = (item.value, item.source, item.evidence.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _skill_buckets(skill_canonicals: set[str], ontology: SkillsOntology) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "language": [],
        "database": [],
        "framework": [],
        "cloud": [],
    }
    for canonical in sorted(skill_canonicals):
        entry = ontology.get(canonical)
        if entry is not None and entry.category in buckets:
            buckets[entry.category].append(canonical)
    return buckets


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value.replace(",", "."))
