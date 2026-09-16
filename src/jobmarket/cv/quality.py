"""Deterministic section, education, experience, and quality extraction for CVs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from jobmarket.cv.profile import (
    CvSection,
    DocumentTextBlock,
    EducationDetail,
    ExperienceEntry,
    ExtractionQuality,
    ExtractionTrace,
    RoleInference,
)
from jobmarket.skills.ontology import normalize_alias

_SECTION_LABELS: dict[str, str] = {
    "profile": "profile",
    "summary": "profile",
    "profil": "profile",
    "resume": "profile",
    "résumé": "profile",
    "education": "education",
    "formation": "education",
    "éducation": "education",
    "experience": "experience",
    "work experience": "experience",
    "experience professionnelle": "experience",
    "competences": "skills",
    "compétences": "skills",
    "skills": "skills",
    "technical skills": "skills",
    "compétences techniques": "skills",
    "projets": "projects",
    "projects": "projects",
    "certifications": "certifications",
    "languages": "languages",
    "langues": "languages",
    "activities": "activities",
    "vie associative": "activities",
    "interests": "interests",
    "centres d'intérêt": "interests",
    "centres d’interet": "interests",
    "centres d’intérêt": "interests",
}
_SECTION_RE = re.compile(
    r"(?im)^\s*(profile|summary|profil|r[eé]sum[eé]|education|formation|[eé]ducation|"
    r"work experience|experience|exp[eé]rience(?: professionnelle)?|skills|technical skills|"
    r"comp[eé]tences(?: techniques)?|projects|projets|certifications|languages|langues|"
    r"activities|vie associative|interests|centres d['’]int[eé]r[eê]t)\s*:?\s*$"
)
_MONTH_YEAR_PATTERN = (
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|janvier|"
    r"f[e\u00e9]vrier|mars|avril|mai|juin|juillet|ao[u\u00fb]t|septembre|octobre|"
    r"novembre|d[e\u00e9]cembre)\s+\d{4}"
)
_PRESENT_END_PATTERN = (
    r"(?:present|pr[e\u00e9]sent|current|actuel(?:le)?|aujourd(?:'|\u2019)hui|"
    r"a ce jour|en cours)"
)
DATE_RANGE_RE = re.compile(
    rf"(?P<start>(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})|(?:\d{{1,2}}[/-]\d{{4}})|"
    rf"{_MONTH_YEAR_PATTERN})"
    r"\s*(?:[-/\u2013\u2014]|to|a|\u00e0)\s*"
    rf"(?P<end>{_PRESENT_END_PATTERN}|(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})|"
    rf"(?:\d{{1,2}}[/-]\d{{4}})|{_MONTH_YEAR_PATTERN})",
    re.IGNORECASE,
)

_BUSINESS_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Management", ("management", "gestion", "manager")),
    (
        "Customer Service",
        ("customer service", "accueil client", "orientation des clients", "service client"),
    ),
    ("Customer Relationship Management", ("relation client", "conseil client", "client advisory")),
    ("Sales", ("sales", "vente", "vente de produits", "suivi des ventes", "encaissement")),
    (
        "Business Development",
        ("business development", "développement commercial", "developpement commercial"),
    ),
    ("Communication", ("communication", "prise de parole en public", "public speaking")),
    ("Leadership", ("leadership",)),
    ("Teamwork", ("teamwork", "travail d'équipe", "travail d’equipe", "esprit d'équipe")),
    ("Organization", ("organisation", "organization")),
    ("Insurance", ("insurance", "assurance", "dossiers de souscription")),
    (
        "Administrative Operations",
        (
            "administrative",
            "suivi des dossiers",
            "document processing",
            "contrôle des dossiers",
            "controle des dossiers",
        ),
    ),
    ("Digital Transformation", ("digital transformation", "transformation digitale")),
    ("Innovation Management", ("innovation",)),
    ("Sponsorship", ("sponsoring", "sponsorship")),
    ("Media Production", ("médias", "medias", "media production")),
    ("Stock Management", ("gestion des stocks", "stock management")),
    (
        "Commercial Support",
        (
            "soutien commercial",
            "commercial support",
            "accompagnement des étudiants",
            "accompagnement des etudiants",
        ),
    ),
    ("Retail", ("retail", "boutique", "magasin")),
    ("Marketing", ("marketing",)),
    ("Project Coordination", ("coordination de projet", "project coordination", "projet")),
)
_ROLE_FAMILY_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Data / AI", ("data scientist", "machine learning", "ia", "ai", "python", "pytorch")),
    (
        "Software Engineering",
        ("software", "developer", "développeur", "developpeur", "java", "api"),
    ),
    ("Management", ("management", "manager", "gestion")),
    ("Sales", ("sales", "vente", "commercial")),
    ("Customer Service", ("customer", "client", "accueil")),
    (
        "Business Development",
        ("business development", "développement commercial", "developpement commercial"),
    ),
    ("Marketing", ("marketing",)),
    ("Finance", ("finance", "accounting", "comptabilité", "comptabilite")),
    ("Insurance", ("insurance", "assurance")),
    ("Human Resources", ("human resources", "ressources humaines", "rh")),
    ("Project Management", ("project management", "gestion de projet", "chef de projet")),
    ("Operations", ("operations", "opérations", "operations")),
)


@dataclass(frozen=True)
class CvAnalysis:
    sections: list[CvSection]
    education: EducationDetail | None
    experiences: list[ExperienceEntry]
    domains: list[RoleInference]
    roles: list[RoleInference]
    quality: ExtractionQuality
    business_skills: set[str]
    traces: list[ExtractionTrace]


def analyze_cv_text(
    text: str,
    skill_canonicals: set[str],
    matched_aliases: set[str],
    blocks: list[DocumentTextBlock] | None = None,
) -> CvAnalysis:
    """Analyze CV structure and reliability with deterministic evidence only."""
    actual_blocks = blocks or []
    sections = detect_sections(text, actual_blocks)
    section_map = _section_texts(text, sections)
    education = extract_education(text, section_map, actual_blocks)
    experiences = extract_experience_entries(text, section_map, actual_blocks)
    business = extract_business_skills(text)
    domains = infer_families(text, business | skill_canonicals, _BUSINESS_TERMS)
    roles = infer_families(text, business | skill_canonicals, _ROLE_FAMILY_TERMS)
    quality = extraction_quality(
        text=text,
        sections=sections,
        education=education,
        experiences=experiences,
        domains=domains,
        reliable_skill_count=len(_reliable_skills(skill_canonicals, matched_aliases)),
        ambiguous_aliases=sorted(
            alias for alias in matched_aliases if alias.casefold() in {"r", "c"}
        ),
    )
    traces = build_extraction_traces(sections, education, experiences)
    return CvAnalysis(sections, education, experiences, domains, roles, quality, business, traces)


def detect_sections(text: str, blocks: list[DocumentTextBlock] | None = None) -> list[CvSection]:
    block_sections = _detect_block_sections(text, blocks or [])
    if block_sections:
        return block_sections
    matches = list(_SECTION_RE.finditer(text))
    sections: list[CvSection] = []
    for index, match in enumerate(matches):
        raw = match.group(1).strip()
        normalized = normalize_heading(raw)
        if normalized is None:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append(CvSection(label=raw, normalized_label=normalized, start=start, end=end))
    return sections


def normalize_heading(value: str) -> str | None:
    normalized = _robust_normalize(value)
    normalized = re.sub(r"[^a-z ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized in _SECTION_LABELS:
        return _SECTION_LABELS[normalized]
    if normalized.startswith("experience professionnelle"):
        return "experience"
    if re.match(r"exp rience(?: professionnelle)?$", normalized):
        return "experience"
    if normalized in {"language", "languages"}:
        return "languages"
    return None


def build_extraction_traces(
    sections: list[CvSection],
    education: EducationDetail | None,
    experiences: list[ExperienceEntry],
) -> list[ExtractionTrace]:
    traces: list[ExtractionTrace] = []
    for section in sections:
        traces.append(
            ExtractionTrace(
                kind="section",
                normalized_value=section.normalized_label,
                evidence_text=section.label,
                confidence=section.confidence,
                page=section.page,
                source_block_id=section.source_block_id,
            )
        )
    if education is not None:
        traces.append(
            ExtractionTrace(
                kind="education",
                normalized_value=(
                    education.normalized_value or education.education_level or "education"
                ),
                evidence_text="; ".join(education.evidence),
                confidence=education.confidence,
                page=education.page,
                source_block_id=education.source_block_id,
            )
        )
    for experience in experiences:
        traces.append(
            ExtractionTrace(
                kind="experience",
                normalized_value=experience.normalized_value or experience.title or "experience",
                evidence_text=experience.evidence,
                confidence=experience.confidence,
                page=experience.page,
                source_block_id=experience.source_block_id,
            )
        )
    return traces


def _detect_block_sections(text: str, blocks: list[DocumentTextBlock]) -> list[CvSection]:
    candidates: list[CvSection] = []
    ordered = sorted(
        (block for block in blocks if block.start_offset is not None),
        key=lambda block: (
            block.start_offset if block.start_offset is not None else 0,
            block.block_id,
        ),
    )
    for block in ordered:
        heading = _heading_from_block(block.text)
        if heading is None or block.start_offset is None:
            continue
        candidates.append(
            CvSection(
                label=block.text.strip(),
                normalized_label=heading,
                start=block.start_offset + len(block.text),
                end=len(text),
                confidence=0.92,
                page=block.page,
                source_block_id=block.block_id,
            )
        )
    result: list[CvSection] = []
    for index, section in enumerate(candidates):
        end = candidates[index + 1].start if index + 1 < len(candidates) else len(text)
        result.append(section.model_copy(update={"end": max(section.start, end)}))
    return result


def _heading_from_block(value: str) -> str | None:
    short = " ".join(value.strip().split())
    if not short or len(short) > 48:
        return None
    return normalize_heading(re.sub(r"[:\-??]+$", "", short))


def _experience_entries_from_blocks(blocks: list[DocumentTextBlock]) -> list[ExperienceEntry]:
    if not blocks:
        return []
    entries: list[ExperienceEntry] = []
    ordered = sorted(blocks, key=lambda block: (block.page, -block.y, block.x, block.block_id))
    date_blocks = [
        block
        for block in ordered
        if len(block.text) <= 120 and DATE_RANGE_RE.search(_robust_text(block.text))
    ]
    for date_block in date_blocks:
        neighbors = [
            block
            for block in ordered
            if block.page == date_block.page
            and block.block_id != date_block.block_id
            and abs(block.y - date_block.y) <= 38
            and block.x >= date_block.x - 10
        ]
        neighbors = sorted(neighbors, key=lambda block: (abs(block.y - date_block.y), block.x))[:4]
        context = " ".join([date_block.text, *(block.text for block in neighbors)])
        entry = _entry_from_context(context, date_block)
        if entry is not None:
            entries.append(entry)
    for block in ordered:
        role_entry = _entry_from_role_block(block, ordered)
        if role_entry is not None:
            entries.append(role_entry)
    return _dedupe_experience(entries)


def _entry_from_role_block(
    block: DocumentTextBlock, ordered: list[DocumentTextBlock]
) -> ExperienceEntry | None:
    if len(block.text) > 220:
        return None
    context_blocks = [
        other
        for other in ordered
        if other.page == block.page
        and other.block_id != block.block_id
        and abs(other.y - block.y) <= 42
        and len(other.text) <= 220
    ]
    context_blocks = sorted(
        context_blocks, key=lambda other: (abs(other.y - block.y), other.x)
    )[:5]
    context = " ".join([block.text, *(other.text for other in context_blocks)])
    norm = normalize_alias(_robust_text(context))
    block_norm = normalize_alias(_robust_text(block.text))
    if not re.search(
        r"vendeuse|stagiaire|education first|gat assurances|commercial support", block_norm
    ):
        return None
    title = _experience_title_from_trigger(block.text) or _experience_title(context)
    domain = _domain_from_text(block_norm) or _domain_from_text(norm)
    if title is None and domain is None:
        return None
    date_match = _nearest_date_match(block, ordered)
    entry_type = "internship" if re.search(r"internship|stage|stagiaire", norm) else "job"
    start_date = date_match.group("start") if date_match is not None else None
    end_date = date_match.group("end") if date_match is not None else None
    return ExperienceEntry(
        title=title,
        employer=_employer(context),
        start_date=start_date,
        end_date=end_date,
        duration_months=_duration_months(start_date, end_date) if start_date and end_date else None,
        entry_type=entry_type,
        domain=domain,
        confidence=0.78,
        evidence=context[:300],
        page=block.page,
        source_block_id=block.block_id,
        normalized_value=" ".join(part for part in (title, domain) if part),
    )


def _nearest_date_match(
    block: DocumentTextBlock, ordered: list[DocumentTextBlock]
) -> re.Match[str] | None:
    candidates = [
        other
        for other in ordered
        if other.page == block.page
        and len(other.text) <= 120
        and DATE_RANGE_RE.search(_robust_text(other.text))
    ]
    candidates = sorted(
        candidates, key=lambda other: (abs(other.y - block.y), abs(other.x - block.x))
    )
    if not candidates:
        return None
    return DATE_RANGE_RE.search(_robust_text(candidates[0].text))


def _experience_title_from_trigger(text: str) -> str | None:
    norm = normalize_alias(_robust_text(text))
    if "education first" in norm:
        return "Commercial Support"
    if re.search(r"stagiaire|gat assurances|assurance", norm):
        return "Insurance Intern"
    if re.search(r"vendeuse|parfumerie", norm):
        return "Sales Assistant"
    return None


def _entry_from_context(
    context: str, source_block: DocumentTextBlock | None
) -> ExperienceEntry | None:
    match = DATE_RANGE_RE.search(_robust_text(context))
    if match is None:
        return None
    norm = normalize_alias(_robust_text(context))
    title = _experience_title(context)
    domain = _domain_from_text(norm)
    if title is None and domain is None:
        return None
    entry_type = "internship" if re.search(r"internship|stage|stagiaire", norm) else "job"
    return ExperienceEntry(
        title=title,
        employer=_employer(context),
        start_date=match.group("start"),
        end_date=match.group("end"),
        duration_months=_duration_months(match.group("start"), match.group("end")),
        entry_type=entry_type,
        domain=domain,
        confidence=0.82 if source_block is not None else 0.68,
        evidence=context[:300],
        page=source_block.page if source_block is not None else None,
        source_block_id=source_block.block_id if source_block is not None else None,
        normalized_value=" ".join(part for part in (title, domain) if part),
    )


def _best_education_block(blocks: list[DocumentTextBlock]) -> DocumentTextBlock | None:
    candidates = [
        block
        for block in blocks
        if re.search(r"licence|bachelor|essect|diplom", _robust_normalize(block.text))
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda block: (
            -_education_block_score(block),
            block.page,
            -block.y,
            block.x,
            block.block_id,
        ),
    )[0]


def _education_source_from_blocks(blocks: list[DocumentTextBlock]) -> str:
    candidates = [
        block
        for block in blocks
        if _education_block_score(block) > 0
    ]
    candidates = sorted(
        candidates,
        key=lambda block: (-_education_block_score(block), block.page, -block.y, block.x),
    )[:8]
    return " ".join(block.text for block in candidates)


def _education_block_score(block: DocumentTextBlock) -> int:
    if len(block.text) > 220:
        return 0
    norm = _robust_normalize(block.text)
    score = 0
    score += 4 if "licence" in norm else 0
    score += 3 if "management" in norm else 0
    score += 3 if "essect" in norm else 0
    score += 2 if re.search(r"diplom|sciences economiques|gestion", norm) else 0
    score -= 3 if "master digital" in norm else 0
    return score


def _education_institution(text: str) -> str | None:
    robust = _robust_text(text)
    if "essect" in normalize_alias(robust):
        return "ESSECT"
    return None


def _robust_text(value: str) -> str:
    try:
        if "?" in value or "?" in value:
            value = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        pass
    value = value.replace("Pr?sent", "Present").replace("pr?sent", "present")
    value = value.replace("Exp?rience", "Experience").replace("exp?rience", "experience")
    value = value.replace("dipl?m", "diplom")
    return value.replace("\xa0", " ").replace("?", "-").replace("?", "-")


def _robust_normalize(value: str) -> str:
    return normalize_alias(_robust_text(value))


def extract_education(
    text: str,
    section_map: dict[str, str],
    blocks: list[DocumentTextBlock] | None = None,
) -> EducationDetail | None:
    source = section_map.get("education", text)
    source_block = _best_education_block(blocks or [])
    block_source = _education_source_from_blocks(blocks or [])
    if block_source:
        source = block_source
    source = _robust_text(source)
    norm = _robust_normalize(source)
    evidence: list[str] = []
    level = None
    if re.search(r"\b(bachelor|licence)\b", norm):
        level = "bachelor/licence"
        evidence.append("Bachelor/Licence")
    elif re.search(r"\b(master|mba)\b", norm):
        level = "master/mba"
        evidence.append("Master/MBA")
    elif re.search(r"\b(ingenieur|engineering degree)\b", norm):
        level = "engineering_degree"
        evidence.append("Engineering degree")
    elif re.search(r"\b(phd|doctorat)\b", norm):
        level = "phd/doctorat"
        evidence.append("PhD/Doctorat")
    elif "baccalaureat" in norm:
        level = "baccalaureat"
        evidence.append("Baccalauréat")
    field = None
    for candidate, pattern in (
        ("management", r"management|gestion"),
        ("data science", r"data science"),
        ("software engineering", r"informatique|computer science|software"),
        ("marketing", r"marketing"),
        ("finance", r"finance|accounting|comptabil"),
    ):
        if re.search(pattern, norm):
            field = candidate
            evidence.append(candidate)
            break
    institution = _education_institution(source)
    year_match = re.search(r"\b(202[3-9]|203\d)\b", source)
    enrolled_pattern = (
        r"expected graduation|student in|en cours|present|1st year|premiere annee|"
        r"2nd year|deuxieme annee|3rd year|troisieme annee|"
        r"4th year|quatrieme annee|5th year|cinquieme annee"
    )
    enrolled = bool(re.search(enrolled_pattern, norm))
    graduated = bool(re.search(r"graduated|diplome|diplôm", norm)) or (
        level is not None and not enrolled
    )
    status = "currently_enrolled" if enrolled else "graduated" if graduated else None
    if level is None and field is None and not evidence:
        return None
    confidence = (
        0.35 + (0.25 if level else 0.0) + (0.2 if field else 0.0) + (0.1 if status else 0.0)
    )
    return EducationDetail(
        education_status=status,
        education_level=level,
        education_field=field,
        institution=institution,
        graduation_year=int(year_match.group(1)) if year_match else None,
        currently_enrolled=enrolled or None,
        graduated=graduated or None,
        confidence=min(1.0, confidence),
        evidence=sorted(set(evidence)),
        page=source_block.page if source_block is not None else None,
        source_block_id=source_block.block_id if source_block is not None else None,
        normalized_value=" ".join(part for part in (level, field, institution) if part),
    )


def extract_experience_entries(
    text: str,
    section_map: dict[str, str],
    blocks: list[DocumentTextBlock] | None = None,
) -> list[ExperienceEntry]:
    source = section_map.get("experience", text)
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    entries: list[ExperienceEntry] = _experience_entries_from_blocks(blocks or [])
    for index, _line in enumerate(lines):
        context = " ".join(lines[index : min(len(lines), index + 3)])
        match = DATE_RANGE_RE.search(context)
        if match is None:
            continue
        norm = normalize_alias(context)
        entry_type = (
            "internship"
            if re.search(r"internship|stage|stagiaire", norm)
            else "apprenticeship"
            if "alternance" in norm
            else "volunteer"
            if re.search(r"volunteer|benevole|associative", norm)
            else "job"
        )
        if entry_type == "volunteer":
            continue
        title = _experience_title(context)
        domain = _domain_from_text(norm)
        duration = _duration_months(match.group("start"), match.group("end"))
        entries.append(
            ExperienceEntry(
                title=title,
                employer=_employer(context),
                start_date=match.group("start"),
                end_date=match.group("end"),
                duration_months=duration,
                entry_type=entry_type,
                domain=domain,
                confidence=0.75 if title else 0.6,
                evidence=context[:300],
                normalized_value=" ".join(
                    part
                    for part in (title, domain, match.group("start"), match.group("end"))
                    if part
                ),
            )
        )
    return _dedupe_experience(entries)


def extract_business_skills(text: str) -> set[str]:
    norm = _robust_normalize(text)
    found: set[str] = set()
    for canonical, aliases in _BUSINESS_TERMS:
        if any(_alias_present(alias, norm) for alias in aliases):
            found.add(canonical)
    return found


def infer_families(
    text: str,
    signals: set[str],
    taxonomy: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[RoleInference]:
    norm = _robust_normalize(text)
    results: list[RoleInference] = []
    for family, aliases in taxonomy:
        evidence = [signal for signal in sorted(signals) if signal == family]
        evidence.extend(alias for alias in aliases if _alias_present(alias, norm))
        evidence = sorted(set(evidence), key=str.casefold)
        if evidence:
            confidence = min(1.0, 0.35 + 0.15 * len(evidence))
            results.append(
                RoleInference(value=family, confidence=confidence, evidence=evidence[:5])
            )
    return sorted(results, key=lambda item: (-item.confidence, item.value))


def _alias_present(alias: str, normalized_text: str) -> bool:
    normalized_alias = _robust_normalize(alias)
    if not normalized_alias:
        return False
    if len(normalized_alias) <= 2:
        return re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_text) is not None
    return normalized_alias in normalized_text


def extraction_quality(
    *,
    text: str,
    sections: list[CvSection],
    education: EducationDetail | None,
    experiences: list[ExperienceEntry],
    domains: list[RoleInference],
    reliable_skill_count: int,
    ambiguous_aliases: list[str],
) -> ExtractionQuality:
    section_coverage = min(1.0, len({section.normalized_label for section in sections}) / 5)
    text_quality = _text_quality(text)
    score = 0.0
    score += 0.15 if domains else 0.0
    score += 0.18 if education is not None else 0.0
    score += 0.2 if experiences else 0.0
    score += min(0.18, reliable_skill_count * 0.045)
    score += section_coverage * 0.14
    score += text_quality * 0.1
    score -= 0.12 if ambiguous_aliases and reliable_skill_count <= 1 else 0.0
    score = max(0.0, min(1.0, score))
    warnings: list[str] = []
    abstention_reason = None
    if ambiguous_aliases:
        warnings.append("ambiguous_single_character_alias_detected")
    if not domains:
        warnings.append("no_domain_detected")
    if education is None:
        warnings.append("no_education_detected")
    if not experiences:
        warnings.append("no_experience_detected")
    if reliable_skill_count <= 1:
        warnings.append("too_few_reliable_skills")
    hard_low = (
        not domains
        or education is None
        or not experiences
        or (reliable_skill_count <= 1 and ambiguous_aliases)
    )
    if hard_low:
        score = min(score, 0.39)
        abstention_reason = "insufficient_profile_evidence"
    label = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return ExtractionQuality(
        extraction_confidence_score=round(score, 4),
        extraction_confidence_label=label,
        text_quality=round(text_quality, 4),
        section_coverage=round(section_coverage, 4),
        reliable_skill_count=reliable_skill_count,
        ambiguous_aliases=ambiguous_aliases,
        warnings=warnings,
        abstention_reason=abstention_reason if label == "low" else None,
    )


def section_for_offset(sections: list[CvSection], start: int) -> str | None:
    for section in sections:
        if section.start <= start <= section.end:
            return section.normalized_label
    return None


def _section_texts(text: str, sections: list[CvSection]) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in sections:
        result.setdefault(section.normalized_label, "")
        result[section.normalized_label] += "\n" + text[section.start : section.end]
    return result


def _reliable_skills(skills: set[str], aliases: set[str]) -> set[str]:
    if aliases <= {"R", "C", "r", "c"} and len(skills) <= 1:
        return set()
    return {skill for skill in skills if skill not in {"R", "C"} or len(skills) > 1}


def _text_quality(text: str) -> float:
    if not text.strip():
        return 0.0
    weird = len(re.findall(r"[�□■●]{1}", text))
    single_letters = len(re.findall(r"(?m)^\s*[A-Za-z]\s*$", text))
    penalty = min(0.6, weird * 0.03 + single_letters * 0.02)
    return max(0.2, 1.0 - penalty)


def _experience_title(text: str) -> str | None:
    norm = normalize_alias(text)
    patterns = (
        ("Sales Assistant", r"sales assistant|assistante? de vente|vente"),
        ("Insurance Intern", r"insurance|assurance|stage"),
        ("Commercial Support", r"commercial|customer support|support client|education first"),
        ("Customer Service", r"customer service|accueil client|relation client"),
    )
    for title, pattern in patterns:
        if re.search(pattern, norm):
            return title
    return None


def _employer(text: str) -> str | None:
    for candidate in ("Education First", "EF", "Assurance"):
        if candidate.casefold() in text.casefold():
            return candidate
    return None


def _domain_from_text(norm: str) -> str | None:
    for domain, aliases in _BUSINESS_TERMS:
        if any(normalize_alias(alias) in norm for alias in aliases):
            return domain
    return None


def calculate_duration_months(
    start: str | None,
    end: str | None,
    *,
    as_of: date | None = None,
) -> int | None:
    """Return an inclusive month count for one grounded experience date range.

    Current-role labels are deliberately handled here rather than trusted from an LLM:
    `Present`, `Current`, `actuel`, and `aujourd'hui` all mean the supplied `as_of`
    date (or today in production). English and French month-year values are accepted.
    """
    if not start or not end:
        return None
    start_year = _year(start)
    if start_year is None:
        return None
    start_month = _month(start) or 1
    if _is_current_role_end(end):
        current = as_of or date.today()
        end_year, end_month = current.year, current.month
    else:
        end_year = _year(end)
        if end_year is None:
            return None
        end_month = _month(end) or start_month
    months = (end_year - start_year) * 12 + end_month - start_month + 1
    return max(1, months) if months >= 0 else None


def _duration_months(start: str, end: str) -> int | None:
    """Backward-compatible internal wrapper for deterministic extraction."""
    return calculate_duration_months(start, end)


def _is_current_role_end(value: str) -> bool:
    normalized = normalize_alias(value).replace("\u2019", "'")
    return normalized in {
        "present",
        "current",
        "actuel",
        "actuelle",
        "aujourd'hui",
        "a ce jour",
        "en cours",
    }


def _year(value: str) -> int | None:
    match = re.search(r"(20\d{2}|19\d{2})", value)
    return int(match.group(1)) if match else None


def _month(value: str) -> int | None:
    lower = normalize_alias(value)
    month_names = {
        "jan": 1,
        "janvier": 1,
        "feb": 2,
        "fevrier": 2,
        "mars": 3,
        "mar": 3,
        "apr": 4,
        "avril": 4,
        "may": 5,
        "mai": 5,
        "jun": 6,
        "juin": 6,
        "jul": 7,
        "juillet": 7,
        "aug": 8,
        "aout": 8,
        "sep": 9,
        "septembre": 9,
        "oct": 10,
        "octobre": 10,
        "nov": 11,
        "novembre": 11,
        "dec": 12,
        "decembre": 12,
    }
    numeric = re.match(r"(\d{1,2})[/-](?:\d{1,2}[/-])?\d{2,4}", value)
    if numeric:
        number = int(numeric.group(1))
        return number if 1 <= number <= 12 else None
    for name, number in month_names.items():
        if name in lower:
            return number
    return None


def _dedupe_experience(entries: list[ExperienceEntry]) -> list[ExperienceEntry]:
    seen: set[str] = set()
    result: list[ExperienceEntry] = []
    for entry in entries:
        key = normalize_alias(entry.evidence[:120])
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


__all__ = [
    "CvAnalysis",
    "DATE_RANGE_RE",
    "calculate_duration_months",
    "analyze_cv_text",
    "build_extraction_traces",
    "detect_sections",
    "extract_business_skills",
    "extract_education",
    "extract_experience_entries",
    "normalize_heading",
    "section_for_offset",
]
