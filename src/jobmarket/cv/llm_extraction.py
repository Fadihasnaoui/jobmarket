"""LLM-based CV profile extraction: one grounded call per CV, ontology-routed skills.

Unlike job postings (Adzuna truncates descriptions to 500 characters), a CV is rich,
mostly-complete text — the case the project's LLM cost/benefit analysis (see
`llm_benchmark.py`) said favors an LLM call. This module makes exactly one LLM call per
CV and asks for everything structured extraction needs: skills, work experience,
internships, education, role families, and seniority.

Skills are never trusted blindly: candidate skill names are only kept if the ontology
resolves them to a canonical entry AND the LLM's evidence quote is an exact substring of
the parsed document (falling back to a same-text case-insensitive lookup of the skill
name itself). The resulting candidates are handed to the existing, unmodified
`cv.llm_guard.validate_llm_skill_candidates` — the same offline grounding/ontology guard
already used for the (previously unused) guarded-candidate path — so skills go through
one single, already-tested validation path regardless of where they came from.

Any failure here (missing API key, network error, malformed JSON after retries, or an
LLM response with no groundable facts at all) raises `CvLlmExtractionError`. Callers are
expected to catch it and fall back to the deterministic parser
(`extract_cv_profile_with_fallback` does exactly that) — this module never silently
returns a low-quality profile instead of raising.

No raw CV text is persisted by this module: the document text is sent to the configured
LLM provider for this one call (as it must be, to be extracted) and otherwise stays in
memory only, matching the existing privacy posture (see `cv/reporting.py`'s
privacy-safe payloads).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import structlog
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field, ValidationError

from jobmarket.config import get_settings
from jobmarket.cv.attributes import extract_candidate_attributes
from jobmarket.cv.extraction import extract_cv_profile
from jobmarket.cv.profile import (
    CandidateAttributes,
    CvProfile,
    EducationDetail,
    ExperienceEntry,
    ExperienceSummary,
    ExtractionVersions,
    ParsedDocument,
    RoleInference,
)
from jobmarket.cv.quality import DATE_RANGE_RE, calculate_duration_months, extraction_quality
from jobmarket.skills.matcher import get_matcher_versions
from jobmarket.skills.ontology import SkillsOntology, load_ontology, normalize_alias

logger = structlog.get_logger(__name__)

CV_LLM_EXTRACTION_VERSION = "cv-llm-extractor-v1"
MAX_CV_CHARS = 6000  # long accented French CVs pushed Groq into echoing raw text back
MAX_ATTEMPTS = 2  # one retry, with a simplified/stricter prompt, before falling back
# No explicit cap was ever passed before, so the client/provider default applied — seen
# in practice to be too low even for a short (~2300 char) input CV: Groq returned a 400
# json_validate_failed with failed_generation="max completion tokens reached before
# generating a valid document" (the model ran out of room mid-JSON on a skill/experience
# -rich profile). One call site (`call_llm_cv_extraction`) serves both the first attempt
# and the simplified retry, so raising this here covers both automatically.
MAX_COMPLETION_TOKENS = 8000

_GROUNDED_EXPERIENCE_CONFIDENCE = 0.75
_TECHNICAL_DOMAINS = {"AI", "Data Science", "Machine Learning", "Software Engineering"}
_TECHNICAL_SKILLS = {"Python", "Data Science", "Machine Learning", "Artificial Intelligence"}
_EDUCATION_LEVEL_RANK = {
    "baccalaureat": 0,
    "bachelor/licence": 1,
    "master/mba": 2,
    "engineering_degree": 2,
    "phd/doctorat": 3,
}

_SYSTEM_PROMPT = (
    "You extract structured facts from one candidate CV/resume for a job-matching "
    "pipeline. Only report what is explicitly stated in the text; never invent, guess, "
    "or infer beyond what's written. Evidence quotes must be exact, verbatim substrings "
    "copied from the CV text — do not paraphrase them. The CV may be in French or "
    'English; recognize French idioms ("confirmé" ~= mid, "stagiaire"/"stage" ~= '
    'internship, "diplômé" ~= graduated, "en cours"/"étudiant" ~= currently enrolled). '
    "Respond with a single strict JSON object matching the given schema exactly — "
    "no prose, no markdown fences, and never repeat or echo the CV text itself back "
    "in the response outside of short evidence_quote fields."
)

# Used only on the retry attempt: shorter and more directive, aimed specifically at
# the failure mode seen in practice (Groq's json_object mode rejecting the response
# with a 400 json_validate_failed, or the model free-typing CV text instead of JSON).
_SIMPLIFIED_SYSTEM_PROMPT = (
    "Output ONLY one JSON object matching the schema below. No prose, no markdown, "
    "no explanations, and do not repeat the CV text back — only short exact quotes in "
    "evidence_quote fields. If unsure about a field, use null or an empty array."
)


_EXPERIENCE_RULES = (
    "Rules for the \"experiences\" array — follow all of them exactly:\n"
    "1. Emit exactly ONE entry per real job or internship the candidate actually held. "
    "Never split one job into multiple entries.\n"
    "2. Bullet points (lines starting with • or ●, or any line describing a "
    "task, achievement, or responsibility) describe the MOST RECENTLY IDENTIFIED job. "
    "They are never a separate job by themselves — never create an experience entry "
    "from a bullet line alone.\n"
    "3. The CV text may list company names, job titles, and date ranges in separate "
    "groups rather than physically next to each other (common when a two-column resume "
    "layout gets flattened to plain text). When this happens, pair the 1st company "
    "found with the 1st job title and 1st date range, the 2nd with the 2nd, and so on, "
    "in the order each group appears — do not leave a title null just because it is not "
    "adjacent to its company in the raw text. Only pair a company with a title or date "
    "range that is genuinely written in the CV — never invent one, and never reuse a "
    "company description, an office location, or a school/university name as if it were "
    "a separate company.\n"
    "4. Only use a null title when you are confident no title exists anywhere in the "
    "text for that job — never for a bullet, and never before you've checked for a "
    "title group elsewhere in the text as described in rule 3.\n"
    "5. A university, college, or school name belongs to EDUCATION, never to "
    "experiences — never use a school name as a job's company/employer.\n"
    "6. Never include certifications, licenses, awards, or other credentials as a work "
    "experience entry, even if the name looks like a job title (e.g. a line under a "
    "heading like \"OTHER\", \"CERTIFICATIONS\", \"AWARDS\", or \"LICENSES\", such as "
    "\"Principal Data Scientist (PDS)\" listed as a certification, is NOT a job — do "
    "not add anything from that section to experiences).\n"
    "7. The line directly under the candidate's name at the very top of the CV is "
    "usually their professional headline (current title), not a job entry by itself. "
    "If that same title text is repeated again next to the candidate's actual first "
    "job/company, that repetition is still real — count it as that first job's title, "
    "do not skip or merge it away just because you already saw the same words once. In "
    "particular: the very first company name listed after that headline (often right "
    "after a divider line) is a real, separate job — always include it as its own "
    "entry with its own title, never drop it or let it cause every job after it to "
    "shift onto the wrong company.\n"
    "8. Older jobs are often listed compactly, commonly under a heading like "
    "\"PREVIOUS EXPERIENCE\": title, company, and location together with the date "
    "range right after — either all on one line (e.g. \"Junior Programmer, ABC "
    "Company, London, UK 06/2017 – 10/2018\"), or, if PDF layout extraction split the "
    "line up, as a title, then a company+location, then a date, in that same order "
    "but on separate lines. Either way, that leading title belongs to the very next "
    "company and date that follow it. Example: three titles listed together (T1, T2, "
    "T3) followed later by three company+date groups in the same order (C1/D1, C2/D2, "
    "C3/D3) pair as T1–C1–D1, T2–C2–D2, T3–C3–D3, strictly first-to-first — never "
    "leave T1's title null just because C1 is written elsewhere in the text.\n"
    "9. A different recurring layout: a company name (sometimes with its location) on "
    "one line, immediately followed by the job title on the very next line — e.g. "
    "\"Acme Corp, New York, NY\" then \"Senior Analyst\" right below it. That pair "
    "belongs together exactly as written, in that immediate company-then-title order, "
    "even if the same title text already appeared once as the candidate's headline at "
    "the top of the CV (see rule 7) — a repeated title next to its own company line is "
    "still that job's real title, not a duplicate to discard. When several such "
    "company/title pairs appear one after another, keep each pair in its own original "
    "position — do not shift a title down onto the next company's line."
)


def _schema_description() -> str:
    return (
        "Return a JSON object with exactly these fields:\n"
        '- "skills": array of {"name": string, "evidence_quote": exact substring from '
        'the CV that names the skill, "confidence": number 0..1}\n'
        '- "experiences": array of {"title": string|null, "company": string|null, '
        '"start_date": string|null, "end_date": string|null, '
        '"duration_months": integer|null, "is_internship": boolean, '
        '"domain": string|null, "evidence_quote": exact substring covering this entry, '
        '"impact_evidence_quote": exact achievement/responsibility quote for this entry or ""}\n'
        '- "education": array of {"level": one of "baccalaureat", "bachelor/licence", '
        '"master/mba", "engineering_degree", "phd/doctorat", or null, "field": '
        'string|null, "institution": string|null, "graduation_year": integer|null, '
        '"status": "graduated" or "currently_enrolled" or null, '
        '"evidence_quote": string|null}\n'
        '- "role_families": array of strings, e.g. "Data / AI", "Software Engineering", '
        '"Sales", "Management"\n'
        '- "domains": array of strings, e.g. "Data Science", "Machine Learning", '
        '"Management", "Customer Service"\n'
        '- "seniority": one of "student", "internship", "junior", "mid", "senior", '
        '"lead", "unknown"\n'
        '- "total_years_experience": number or null\n'
        "Only include a skill, experience, or education entry if you can point to an "
        "exact quote for it in the CV text.\n\n" + _EXPERIENCE_RULES
    )


def _build_user_prompt(text: str) -> str:
    return f"CV text:\n{text}\n\n{_schema_description()}"


def _build_messages(text: str, *, simplified: bool) -> list[dict[str, str]]:
    if simplified:
        return [
            {"role": "system", "content": _SIMPLIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": f"CV text:\n{text}\n\n{_schema_description()}"},
        ]
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text)},
    ]


class CvLlmExtractionError(RuntimeError):
    """Raised when LLM-based CV extraction cannot produce a usable profile.

    Callers should catch this and fall back to the deterministic parser.
    """


class LlmSkillCandidate(BaseModel):
    model_config = {"extra": "ignore"}
    name: str
    evidence_quote: str = ""
    confidence: float = 0.7


class LlmExperienceEntry(BaseModel):
    model_config = {"extra": "ignore"}
    title: str | None = None
    company: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int | None = None
    is_internship: bool = False
    domain: str | None = None
    evidence_quote: str = ""
    impact_evidence_quote: str = ""


EducationLevel = Literal[
    "baccalaureat", "bachelor/licence", "master/mba", "engineering_degree", "phd/doctorat"
]
EducationStatus = Literal["graduated", "currently_enrolled"]


class LlmEducationEntry(BaseModel):
    model_config = {"extra": "ignore"}
    level: EducationLevel | None = None
    field: str | None = None
    institution: str | None = None
    graduation_year: int | None = None
    status: EducationStatus | None = None
    evidence_quote: str | None = None


Seniority = Literal["student", "internship", "junior", "mid", "senior", "lead", "unknown"]


class LlmCvExtraction(BaseModel):
    model_config = {"extra": "ignore"}
    skills: list[LlmSkillCandidate] = Field(default_factory=list)
    experiences: list[LlmExperienceEntry] = Field(default_factory=list)
    education: list[LlmEducationEntry] = Field(default_factory=list)
    role_families: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    seniority: Seniority = "unknown"
    total_years_experience: float | None = None


@dataclass(frozen=True)
class LlmCvDraft:
    """Draft CV profile from one LLM call, before the skill-candidate evidence guard runs.

    `profile.skills` contains deterministic exact-text matches. `skill_candidates` is
    the raw `GuardedSkillCandidate`-shaped LLM list to run through
    `cv.llm_guard.validate_llm_skill_candidates`, the single ontology/evidence
    validation path shared with the rest of the CV pipeline.
    """

    profile: CvProfile
    skill_candidates: list[dict[str, Any]]


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.llm_api_key:
        raise CvLlmExtractionError("LLM_API_KEY is not set (see .env.example)")
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def call_llm_cv_extraction(text: str, *, client: OpenAI | None = None) -> LlmCvExtraction:
    """One LLM call producing strict-JSON, Pydantic-validated CV facts.

    Sends `response_format={"type": "json_object"}` (Groq/OpenAI strict JSON mode).
    That alone isn't a guarantee: Groq can still reject a response outright with an
    HTTP 400 `json_validate_failed` (seen in practice on long, accented French CVs
    where the model started echoing raw CV text instead of JSON), separate from the
    model returning content that merely fails to *parse* as valid JSON. Both failure
    modes get exactly one retry with a shorter, more directive "simplified" prompt
    before raising `CvLlmExtractionError` for the caller to fall back deterministically.
    """
    settings = get_settings()
    llm_client = client or _client()
    truncated = text[:MAX_CV_CHARS]
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        messages = _build_messages(truncated, simplified=attempt > 1)
        try:
            response = llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[call-overload]
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
        except BadRequestError as exc:
            last_error = f"bad_request (likely json_validate_failed): {exc}"
            if attempt < MAX_ATTEMPTS:
                continue
            raise CvLlmExtractionError(last_error) from exc
        except Exception as exc:  # network/timeout/rate-limit/auth/sdk errors
            # Not a JSON-shape problem a different prompt would fix — fail fast.
            raise CvLlmExtractionError(f"LLM call failed: {exc}") from exc
        raw = response.choices[0].message.content or ""
        try:
            return LlmCvExtraction.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"invalid JSON: {exc}"
            if attempt < MAX_ATTEMPTS:
                continue
            raise CvLlmExtractionError(
                f"LLM did not return valid JSON after {MAX_ATTEMPTS} attempts: {last_error}"
            ) from exc
    raise CvLlmExtractionError(f"LLM extraction failed after {MAX_ATTEMPTS} attempts: {last_error}")


def extract_cv_profile_llm(
    document: ParsedDocument,
    *,
    ontology: SkillsOntology | None = None,
    skill_ids: dict[str, int] | None = None,
    client: OpenAI | None = None,
) -> LlmCvDraft:
    """Build a draft CV profile from one grounded LLM call.

    Raises `CvLlmExtractionError` on any failure, including a response with nothing
    groundable in it, so the caller always has a clean fallback signal.
    """
    actual_ontology = ontology or load_ontology()
    actual_skill_ids = skill_ids or _default_skill_ids(actual_ontology)
    try:
        extraction = call_llm_cv_extraction(document.text, client=client)

        # The deterministic matcher is an exact-text, ontology-backed path. Keep its
        # grounded matches even when the LLM call succeeds: an LLM omission must not
        # erase a skill that is visibly written in the CV.
        deterministic_profile = extract_cv_profile(
            document, ontology=actual_ontology, skill_ids=actual_skill_ids
        )
        deterministic_skills = deterministic_profile.skills
        skill_candidates = _build_skill_candidates(
            extraction.skills, document, actual_ontology, actual_skill_ids
        )
        experience_entries = _ground_experiences(extraction.experiences, document)
        education_detail = _select_education(extraction.education)
        domains = _role_inferences(extraction.domains)
        roles = _role_inferences(extraction.role_families)

        if not skill_candidates and not experience_entries and education_detail is None:
            raise CvLlmExtractionError("LLM extraction produced no groundable facts")

        skill_names = {
            *(skill.canonical_skill for skill in deterministic_skills),
            *(candidate["canonical_skill"] for candidate in skill_candidates),
        }
        det_candidate = extract_candidate_attributes(document.text, skill_names, actual_ontology)
        candidate = _merge_llm_candidate(
            det_candidate, extraction, experience_entries, education_detail, skill_names
        )
        quality = extraction_quality(
            text=document.text,
            sections=[],
            education=education_detail,
            experiences=experience_entries,
            domains=domains,
            reliable_skill_count=len(skill_names),
            ambiguous_aliases=[],
        )
        versions = get_matcher_versions(actual_ontology)
        profile = CvProfile(
            document=document,
            versions=ExtractionVersions(
                parser_version=document.parser_version,
                matcher_version=versions.matcher_version,
                ontology_version=versions.ontology_version,
            ),
            skills=deterministic_skills,
            candidate=candidate,
            warnings=document.warnings,
            sections=[],
            education_detail=education_detail,
            experience_entries=experience_entries,
            inferred_roles=roles,
            inferred_domains=domains,
            extraction_quality=quality,
            extraction_traces=[],
        )
        return LlmCvDraft(profile=profile, skill_candidates=skill_candidates)
    except CvLlmExtractionError:
        raise
    except Exception as exc:  # any unexpected failure still falls back cleanly
        raise CvLlmExtractionError(f"LLM CV extraction failed: {exc}") from exc


@dataclass(frozen=True)
class CvExtractionResult:
    """Outcome of `extract_cv_profile_with_fallback` — never silently degrades.

    `extraction_method` is `"llm"` or `"deterministic_fallback"`. `fallback_reason` is
    `None` on the `"llm"` path and a human-readable reason whenever the deterministic
    parser had to be used instead — callers must surface it (see `cv/workflow.py` and
    `cv/service.py`) rather than let a fallback look like an ordinary success.
    """

    profile: CvProfile
    skill_candidates: list[dict[str, Any]]
    extraction_method: Literal["llm", "deterministic_fallback"]
    fallback_reason: str | None = None


def extract_cv_profile_with_fallback(
    document: ParsedDocument,
    *,
    ontology: SkillsOntology | None = None,
    skill_ids: dict[str, int] | None = None,
    client: OpenAI | None = None,
) -> CvExtractionResult:
    """Try one-call LLM CV extraction; fall back to the deterministic parser on failure.

    `skill_candidates` is the raw list to guard through `validate_llm_skill_candidates`;
    it is empty for the deterministic fallback, whose `profile.skills` are already final.
    """
    actual_ontology = ontology or load_ontology()
    actual_skill_ids = skill_ids or _default_skill_ids(actual_ontology)
    try:
        draft = extract_cv_profile_llm(
            document, ontology=actual_ontology, skill_ids=actual_skill_ids, client=client
        )
        return CvExtractionResult(draft.profile, draft.skill_candidates, "llm")
    except CvLlmExtractionError as exc:
        reason = str(exc)
        _log_fallback(document.filename, reason)
        fallback_profile = extract_cv_profile(
            document, ontology=actual_ontology, skill_ids=actual_skill_ids
        )
        return CvExtractionResult(fallback_profile, [], "deterministic_fallback", reason)


def _log_fallback(filename: str, reason: str) -> None:
    try:
        logger.info("cv_llm_extraction_fallback", filename=filename, reason=reason)
    except ValueError:
        # Test harnesses may close captured stdout after Typer configures structlog.
        return


def _default_skill_ids(ontology: SkillsOntology) -> dict[str, int]:
    return {entry.canonical: index for index, entry in enumerate(ontology.entries, start=1)}


def _build_skill_candidates(
    items: list[LlmSkillCandidate],
    document: ParsedDocument,
    ontology: SkillsOntology,
    skill_ids: dict[str, int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in items:
        canonical = ontology.resolve(item.name)
        if canonical is None or canonical not in skill_ids:
            continue
        span = _locate_evidence(document.text, item.evidence_quote, item.name)
        if span is None:
            continue
        start, end = span
        candidates.append(
            {
                "fact_type": "skill",
                "canonical_skill": canonical,
                "ontology_skill_id": skill_ids[canonical],
                "evidence_quote": document.text[start:end],
                "document_start": start,
                "document_end": end,
                "page": None,
                "confidence": min(1.0, max(0.0, item.confidence)),
            }
        )
    return candidates


def _locate_evidence(text: str, quote: str, name: str) -> tuple[int, int] | None:
    stripped = quote.strip()
    if stripped:
        start = text.find(stripped)
        if start != -1:
            return start, start + len(stripped)
    lowered = text.casefold()
    start = lowered.find(name.strip().casefold())
    if start != -1:
        return start, start + len(name.strip())
    return None


def _experience_is_grounded(item: LlmExperienceEntry, document_text: str) -> bool:
    """A job is real only if its company is actually in the CV, as a substring.

    Checking `evidence_quote` alone isn't enough: a bullet line the LLM mistook for
    its own job (the phantom-role bug this guards against) always has a "grounded"
    evidence_quote, since the bullet text is itself real CV text — but it has no
    company. A certification pulled in from an OTHER/awards section is the same
    shape: its *name* is real CV text too (so a title-only check would wrongly
    accept it), but it was never listed with a company. Requiring company
    specifically — not title-or-company — rejects both. A real job missing an
    employer name entirely (e.g. an informal "Freelance" entry with no company
    field filled in) would also be dropped by this; that tradeoff is accepted here
    since every real job on the CVs this was tuned against listed a company, and a
    title that only paraphrases the CV (never an exact substring) shouldn't be
    trusted to prove a job by itself. Same substring-grounding approach already
    used for skills (`_locate_evidence`), applied to the company field.
    """
    company = (item.company or "").strip()
    if not company or company.casefold() not in document_text.casefold():
        return False
    return not (_looks_like_institution(company) or _looks_like_business_description(company))


_INSTITUTION_MARKERS = (
    "university",
    "universite",
    "college",
    "institute",
    "institut",
    "polytechnic",
    "school of",
    "faculty",
    "faculte",
)


def _looks_like_institution(company: str) -> bool:
    """Reject a school/university masquerading as an employer.

    Observed failure mode on a real CV: a job title sitting right next to the
    candidate's university name (in the education block) got paired with the
    university as if it were the employer — a wrong pairing, not a phantom, so the
    company-grounding check above doesn't catch it (the university name really is
    in the CV text). This is a generic category error, not specific to one CV:
    education institutions should never end up as a job's employer.
    """
    lowered = company.casefold()
    return any(marker in lowered for marker in _INSTITUTION_MARKERS)


_DESCRIPTIVE_COMPANY_MARKERS = (
    "employees",
    "revenue",
    "users",
    "clients",
    "startup",
    "saas",
    "recruitment",
    "listed",
    "membership",
)


def _looks_like_business_description(company: str) -> bool:
    """Reject a company blurb/location line masquerading as an employer name.

    Observed failure mode on a real CV (non-deterministic — it doesn't reproduce on
    every call at temperature=0): the model occasionally treats a company's *own
    marketing blurb* ("Education technology startup with 50+ employees and $100m+
    annual revenue") or a bare office-location line ("London, United Kingdom") as a
    separate, additional employer — both are real substrings of the CV, so the
    grounding check alone doesn't catch them. A real company name is a short proper
    noun; a multi-word sentence with statistics is not. This is a shape heuristic,
    not a fix for the underlying non-determinism — a short, digit-free location
    string (e.g. "Paris, France" with no company name at all) can still slip through
    undetected; see docs/LIMITATIONS.md.
    """
    if len(company.split()) > 6:
        return True
    if any(char.isdigit() for char in company):
        return True
    lowered = company.casefold()
    return any(marker in lowered for marker in _DESCRIPTIVE_COMPANY_MARKERS)


def _resolve_experience_dates(
    item: LlmExperienceEntry, document_text: str
) -> tuple[str | None, str | None, int | None]:
    """Prefer a nearby, literal CV date range over incomplete LLM date fields.

    The role/company/evidence still must pass the existing grounding checks. This only
    fills or corrects dates from a date range physically near that grounded role, so a
    model omission such as ``start_date=None, end_date='Present'`` cannot hide a
    duration that is plainly written in the CV.
    """
    normalized_text = normalize_alias(document_text).replace("\u2019", "'")
    anchors: list[int] = []
    for raw_anchor in (item.evidence_quote, item.company, item.title):
        anchor = normalize_alias(raw_anchor or "").replace("\u2019", "'")
        if not anchor:
            continue
        anchors.extend(match.start() for match in re.finditer(re.escape(anchor), normalized_text))

    inferred: tuple[str, str] | None = None
    ranges = list(DATE_RANGE_RE.finditer(normalized_text))
    if anchors and ranges:
        nearest = min(
            ranges,
            key=lambda match: min(abs(match.start() - anchor) for anchor in anchors),
        )
        if min(abs(nearest.start() - anchor) for anchor in anchors) <= 500:
            inferred = (nearest.group("start"), nearest.group("end"))

    start_date = inferred[0] if inferred is not None else item.start_date
    end_date = inferred[1] if inferred is not None else item.end_date
    duration = calculate_duration_months(start_date, end_date)
    return start_date, end_date, duration if duration is not None else item.duration_months


def _ground_experiences(
    items: list[LlmExperienceEntry], document: ParsedDocument
) -> list[ExperienceEntry]:
    entries: list[ExperienceEntry] = []
    seen_keys: set[tuple[str, str]] = set()
    for item in items:
        if not _experience_is_grounded(item, document.text):
            continue
        key = ((item.title or "").strip().casefold(), (item.company or "").strip().casefold())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        quote = item.evidence_quote.strip()
        evidence = quote if quote and quote in document.text else _fallback_evidence(item)
        impact_evidence = _ground_impact_evidence(item, document.text)
        start_date, end_date, duration_months = _resolve_experience_dates(item, document.text)
        entries.append(
            ExperienceEntry(
                title=item.title,
                employer=item.company,
                start_date=start_date,
                end_date=end_date,
                duration_months=duration_months,
                entry_type="internship" if item.is_internship else "job",
                domain=item.domain,
                confidence=_GROUNDED_EXPERIENCE_CONFIDENCE,
                evidence=evidence,
                impact_evidence=impact_evidence,
                normalized_value=" ".join(part for part in (item.title, item.domain) if part)
                or None,
            )
        )
    return entries


def _fallback_evidence(item: LlmExperienceEntry) -> str:
    return " ".join(part for part in (item.title, item.company) if part) or "llm_extraction"


def _ground_impact_evidence(item: LlmExperienceEntry, document_text: str) -> str | None:
    """Keep a short exact per-role impact quote only when it is genuinely in the CV.

    The quality report runs after raw CV text is redacted from the short-lived session,
    so this is the grounded fact it needs to assess whether a role has a measurable
    result. An empty or paraphrased LLM value is deliberately discarded.
    """
    quote = item.impact_evidence_quote.strip()
    if not quote:
        return None
    start = document_text.casefold().find(quote.casefold())
    if start == -1:
        return None
    return document_text[start : start + len(quote)]


def _education_rank(item: LlmEducationEntry) -> tuple[int, int]:
    return (_EDUCATION_LEVEL_RANK.get(item.level or "", -1), item.graduation_year or 0)


def _select_education(items: list[LlmEducationEntry]) -> EducationDetail | None:
    if not items:
        return None
    best = max(items, key=_education_rank)
    evidence = [best.evidence_quote] if best.evidence_quote else []
    return EducationDetail(
        education_status=best.status,
        education_level=best.level,
        education_field=best.field,
        institution=best.institution,
        graduation_year=best.graduation_year,
        currently_enrolled=(best.status == "currently_enrolled") or None,
        graduated=(best.status == "graduated") or None,
        confidence=0.7,
        evidence=evidence,
        normalized_value=" ".join(
            part for part in (best.level, best.field, best.institution) if part
        )
        or None,
    )


def _role_inferences(values: list[str]) -> list[RoleInference]:
    seen = sorted({value.strip() for value in values if value and value.strip()})
    return [
        RoleInference(value=value, confidence=0.6, evidence=["llm_extraction"]) for value in seen
    ]


def _merge_llm_candidate(
    det_candidate: CandidateAttributes,
    extraction: LlmCvExtraction,
    experiences: list[ExperienceEntry],
    education: EducationDetail | None,
    skill_names: set[str],
) -> CandidateAttributes:
    # Only count entries whose evidence actually grounded in the CV text — the LLM's
    # own raw entry_type count is not used here, so an over-counted or fabricated
    # experience/internship (ungrounded, confidence < _GROUNDED_EXPERIENCE_CONFIDENCE)
    # can no longer inflate the merged count. max() with the deterministic regex count
    # still lets a true count the LLM missed win, since that's a floor, not a ceiling.
    grounded_internship_count = sum(
        1
        for entry in experiences
        if entry.entry_type == "internship" and entry.confidence >= _GROUNDED_EXPERIENCE_CONFIDENCE
    )
    grounded_professional_count = sum(
        1
        for entry in experiences
        if entry.entry_type == "job" and entry.confidence >= _GROUNDED_EXPERIENCE_CONFIDENCE
    )
    months = [entry.duration_months for entry in experiences if entry.duration_months]
    total_months = (
        sum(months)
        if months
        else (
            round(extraction.total_years_experience * 12)
            if extraction.total_years_experience is not None
            else None
        )
    )
    total_years = (
        round(total_months / 12, 2)
        if total_months is not None
        else extraction.total_years_experience
    )
    experience_summary = ExperienceSummary(
        internship_count=max(
            det_candidate.experience.internship_count, grounded_internship_count
        ),
        professional_experience_count=max(
            det_candidate.experience.professional_experience_count, grounded_professional_count
        ),
        total_months=total_months,
        total_years=total_years,
        evidence=sorted({entry.evidence for entry in experiences if entry.evidence}),
    )
    career_level = (
        extraction.seniority if extraction.seniority != "unknown" else det_candidate.career_level
    )
    current_role = (
        experiences[0].title
        if experiences and experiences[0].title
        else det_candidate.current_role
    )
    domains = _filtered_domains(
        set(extraction.domains) | set(det_candidate.preferred_domains), skill_names
    )
    return det_candidate.model_copy(
        update={
            "current_role": current_role,
            "career_level": career_level,
            "education_status": (
                education.education_status if education else det_candidate.education_status
            ),
            "experience": experience_summary,
            "preferred_domains": sorted(domains),
        }
    )


def _filtered_domains(domains: set[str], skill_names: set[str]) -> set[str]:
    if not skill_names & _TECHNICAL_SKILLS:
        domains -= _TECHNICAL_DOMAINS
    return domains


__all__ = [
    "CV_LLM_EXTRACTION_VERSION",
    "CvExtractionResult",
    "CvLlmExtractionError",
    "LlmCvDraft",
    "LlmCvExtraction",
    "call_llm_cv_extraction",
    "extract_cv_profile_llm",
    "extract_cv_profile_with_fallback",
]
