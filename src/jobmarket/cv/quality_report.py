"""CV Quality Report: a scored diagnostic, not a rewrite.

Extends `cv/improver.py` with a second, independent capability that shares its
grounding discipline and LLM plumbing (`_client`, `CvImproverError`, `_numbers_in`,
`_grounded_skill_names`, `_recommended_skills`) rather than re-deriving it. Where
`generate_improved_cv` rewords real text, this module only *diagnoses* it — five of
its six checks are pure deterministic reads of the already-grounded `CvProfile`;
only spelling/grammar detection makes an LLM call, and even that call only ever
*flags* an issue plus a corrected version — it never gets applied anywhere here.

Every check operates on facts the pipeline already grounded — nothing here re-reads
raw document text or invents a fact the extractor didn't already produce. Two sharpest
edges, both explicit user requirements:
  - Every corrected spelling/grammar suggestion is validated the same way an
    `generate_improved_cv` rewrite is: numbers must match exactly (not just be a
    subset — a spelling fix must not add OR remove a digit), skill mentions must be
    identical, and the edit must be small relative to sentence length. Anything that
    fails any of those checks is dropped, not surfaced.
  - "Missing contact" (`_contact_check`) checks email/phone/LinkedIn-or-GitHub against
    BOTH the caller-supplied `contact_info` field AND `ContactSignals` — three booleans
    pattern-scanned from the CV's own raw text (see `detect_contact_signals`). This
    module still never extracts or stores an actual email/phone/handle from a CV —
    `ContactSignals` only ever answers "was something matching this shape present?",
    never "what was it?", and it must be computed by the caller *before* the CV's raw
    document text is redacted for storage (`api/store.py::redact_document_text`) —
    this module has no access to that text itself once a CV is behind a `cv_id`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Literal

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jobmarket.config import get_settings
from jobmarket.cv.improver import (
    CvImproverError,
    RecommendedSkill,
    SkillGapEntry,
    _client,
    _numbers_in,
    _recommended_skills,
)
from jobmarket.cv.profile import CvProfile
from jobmarket.cv.workflow import RecommendationOutput
from jobmarket.skills.matcher import match_skill_canonicals
from jobmarket.skills.ontology import SkillsOntology, load_ontology

MAX_SPELLING_ATTEMPTS = 2
SPELLING_MAX_COMPLETION_TOKENS = 2000

QualityCheckStatus = Literal["pass", "warning", "fail", "not_checked"]


class SpellingIssue(BaseModel):
    """One accepted spelling/grammar correction — already grounding-checked."""

    model_config = ConfigDict(frozen=True)

    source_label: str
    original: str
    corrected: str


class UnquantifiedBullet(BaseModel):
    """One real experience entry whose evidence contains no measurable result."""

    model_config = ConfigDict(frozen=True)

    source_label: str
    text: str


class QualityCheck(BaseModel):
    """One scored diagnostic category."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    status: QualityCheckStatus
    score: float
    max_score: float
    message: str
    suggestion: str | None = None


_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_CANDIDATE_PATTERN = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{5,}\d)(?!\d)")
_YEAR_RANGE_PATTERN = re.compile(r"^\s*\d{4}\s*[-–—]\s*\d{4}\s*$")
# URL form is the strongest signal; a bare mention of the word is still accepted —
# real-world PDFs routinely lose a hyperlink's actual href during plain-text
# extraction, leaving only the clickable anchor text ("LinkedIn"/"GitHub") behind, so
# requiring the full URL would reproduce the exact false-negative this check exists to
# fix. Deliberate tradeoff: a CV that names "GitHub" only as a tool/skill, never as a
# personal link, would also pass this — accepted, since this is a diagnostic hint, not
# a hard gate, and the alternative (missing a real link most of the time) is worse.
_LINKEDIN_OR_GITHUB_PATTERN = re.compile(
    r"linkedin\.com/\S+|github\.com/\S+|\blinkedin\b|\bgithub\b", re.IGNORECASE
)
_METRIC_PATTERN = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*%|"
    r"\b\d+(?:[.,]\d+)?\s*(?:percent(?:age)?s?|hours?|hrs?|days?|weeks?|months?|"
    r"years?|users?|customers?|clients?|projects?|tickets?|bugs?|errors?|requests?|"
    r"records?|reports?|teams?|people|employees|candidates|applications)\b|"
    r"[$\u20ac\u00a3]\s*\d+(?:[.,]\d+)?\s*(?:k|m|b|thousand|million)?\b|"
    r"\b\d+\+\s*(?:users?|customers?|clients?|projects?|tickets?|bugs?|errors?|"
    r"requests?|records?|reports?|teams?|people|employees|candidates|applications)\b)",
    re.IGNORECASE,
)
_NAME_EXCLUDED_LINES = {
    "experience", "education", "skills", "competences", "competencies", "summary",
    "profile", "contact", "languages", "certifications", "projects", "volunteer",
}
_DESCRIPTIVE_ACTION_PATTERN = re.compile(
    r"\b(?:built|developed|designed|implemented|improved|reduced|increased|managed|led|"
    r"created|delivered|worked|supported|trained|analy[sz]ed|optimi[sz]ed|collaborated|"
    r"automated|volunteered|assisted|maintained|deployed|contributed|utili[sz]ed|"
    r"cree|developpe|concu|mis en place|ameliore|reduit|augmente|gere|dirige|"
    r"accompagne|forme|analyse|optimise|automatise|deploie|contribue)\b",
    re.IGNORECASE,
)


class ContactSignals(BaseModel):
    """Whether email/phone/LinkedIn-or-GitHub patterns were found in a CV's raw text.

    Four booleans, never the matched substrings — this is the one place outside the
    caller-supplied name/contact form fields that looks at a CV's raw text at all, and
    it must not become a new place PII gets persisted. Must be computed by the caller
    at upload time, before `api/store.py::redact_document_text` discards that text;
    see `detect_contact_signals` and the module docstring.
    """

    model_config = ConfigDict(frozen=True)

    has_email: bool = False
    has_phone: bool = False
    has_linkedin_or_github: bool = False
    has_name: bool = False


def _looks_like_phone(candidate: str) -> bool:
    if _YEAR_RANGE_PATTERN.match(candidate):
        return False  # "2020-2023" is a date range, not a phone number.
    digit_count = sum(1 for char in candidate if char.isdigit())
    return 7 <= digit_count <= 15


def detect_contact_signals(text: str) -> ContactSignals:
    """Pattern-scan raw CV text for contact channels — never returns the match itself.

    Callers must run this at upload time, on the freshly parsed document's text,
    before that text is redacted for storage — nothing later has access to it.
    """
    phone_candidates = _PHONE_CANDIDATE_PATTERN.findall(text)
    return ContactSignals(
        has_email=bool(_EMAIL_PATTERN.search(text)),
        has_phone=any(_looks_like_phone(candidate) for candidate in phone_candidates),
        has_linkedin_or_github=bool(_LINKEDIN_OR_GITHUB_PATTERN.search(text)),
        has_name=_has_header_name(text),
    )


def _has_header_name(text: str) -> bool:
    """Recognise a likely name in the first few CV lines without retaining it."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    for line in lines[:6]:
        candidate = re.split(r"\s*[|\u2022\u00b7]\s*", line, maxsplit=1)[0].strip()
        folded = candidate.casefold().rstrip(":")
        if folded in _NAME_EXCLUDED_LINES or any(char.isdigit() for char in candidate):
            continue
        if "@" in candidate or "http" in folded or "linkedin" in folded or "github" in folded:
            continue
        words = re.findall(
            r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff][A-Za-z'\u2019-]*", candidate
        )
        if 2 <= len(words) <= 5 and len(candidate) <= 60:
            return True
    return False


class CvQualityReport(BaseModel):
    """Result of `generate_cv_quality_report`."""

    model_config = ConfigDict(frozen=True)

    overall_score: int
    checks: list[QualityCheck] = Field(default_factory=list)
    spelling_issues: list[SpellingIssue] = Field(default_factory=list)
    unquantified_bullets: list[UnquantifiedBullet] = Field(default_factory=list)
    recommended_skills_to_develop: list[RecommendedSkill] = Field(default_factory=list)
    dropped_spelling_suggestions: list[str] = Field(default_factory=list)
    source_degraded: bool = False
    source_degraded_reason: str | None = None


_SPELLING_SYSTEM_PROMPT = (
    "You proofread real CV text for spelling and grammar mistakes ONLY. You do not "
    "reword for style or impact, you do not add or remove facts, numbers, or skills, "
    "and you do not shorten or expand a sentence beyond fixing the actual mistake. "
    "If a block has no real spelling or grammar mistake, do not include it in your "
    "output at all — do not invent an issue just to have something to report. "
    "Respond with a single strict JSON object matching the given schema exactly — no "
    "prose, no markdown fences."
)

_SPELLING_SIMPLIFIED_PROMPT = (
    "Output ONLY one JSON object matching the schema below. No prose, no markdown. "
    "Only include a block index if it has a real spelling/grammar mistake, and only "
    "fix that mistake — never reword, never change facts/numbers/skills."
)


class _LlmSpellingIssue(BaseModel):
    model_config = {"extra": "ignore"}
    index: int
    corrected: str = ""


class _LlmSpellingResponse(BaseModel):
    model_config = {"extra": "ignore"}
    issues: list[_LlmSpellingIssue] = Field(default_factory=list)


def _spelling_source_blocks(profile: CvProfile) -> list[tuple[str, str]]:
    """(source_label, real evidence text) pairs to proofread — experience only.

    Education's own evidence is typically a short institution/degree fragment, not
    prose worth proofreading — scoped out deliberately, not an oversight.
    """
    blocks: list[tuple[str, str]] = []
    for entry in profile.experience_entries:
        text = (entry.evidence or "").strip()
        if not text:
            continue
        label = f"{entry.title or 'Untitled role'} @ {entry.employer or 'unknown employer'}"
        blocks.append((label, text))
    return blocks


def _build_spelling_messages(
    blocks: list[tuple[str, str]], *, simplified: bool
) -> list[dict[str, str]]:
    lines = ["Proofread each numbered text block below for spelling/grammar mistakes only:\n"]
    for index, (label, text) in enumerate(blocks):
        lines.append(f"[{index}] (context only, not part of the text: {label})")
        lines.append(f"    text: {text}")
    lines.append(
        '\nReturn a JSON object: {"issues": [{"index": int, "corrected": string}, ...]}. '
        "Only include an index if that exact block's text has a real mistake — if it's "
        'already correct, omit it entirely. "corrected" must be ONLY the fixed text '
        "itself — never include the block's label, context note, or index — fixing ONLY "
        "the mistake: same facts, same numbers, same skills, same meaning, minimal "
        "wording change."
    )
    system = _SPELLING_SIMPLIFIED_PROMPT if simplified else _SPELLING_SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _call_llm_spelling_check(
    blocks: list[tuple[str, str]], *, client: OpenAI | None = None
) -> _LlmSpellingResponse:
    settings = get_settings()
    llm_client = client or _client()
    last_error = "unknown error"
    for attempt in range(1, MAX_SPELLING_ATTEMPTS + 1):
        messages = _build_spelling_messages(blocks, simplified=attempt > 1)
        try:
            response = llm_client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[call-overload]
                response_format={"type": "json_object"},
                temperature=0,
                max_completion_tokens=SPELLING_MAX_COMPLETION_TOKENS,
            )
        except BadRequestError as exc:
            last_error = f"bad_request: {exc}"
            if attempt < MAX_SPELLING_ATTEMPTS:
                continue
            raise CvImproverError(last_error) from exc
        except Exception as exc:  # network/timeout/rate-limit/auth/sdk errors
            raise CvImproverError(f"LLM call failed: {exc}") from exc
        raw = response.choices[0].message.content or ""
        try:
            return _LlmSpellingResponse.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"invalid JSON: {exc}"
            if attempt < MAX_SPELLING_ATTEMPTS:
                continue
            raise CvImproverError(f"LLM did not return valid JSON: {last_error}") from exc
    raise CvImproverError(
        f"Spelling check failed after {MAX_SPELLING_ATTEMPTS} attempts: {last_error}"
    )


def _is_disguised_rewrite(original: str, corrected: str) -> bool:
    """A spelling fix should change very little — reject anything shaped like a
    rewrite instead (that's `generate_improved_cv`'s job, not this check's)."""
    original_words = original.split()
    corrected_words = corrected.split()
    allowed_delta = max(3, len(original_words) // 3)
    return abs(len(corrected_words) - len(original_words)) > allowed_delta


def _normalise_comparison_text(text: str) -> str:
    """Ignore invisible/whitespace-only edits; they are not a real correction."""
    return " ".join(text.replace("\u00a0", " ").split())


def _detect_spelling_issues(
    profile: CvProfile,
    ontology: SkillsOntology,
    *,
    client: OpenAI | None = None,
) -> tuple[list[SpellingIssue], list[str], bool]:
    """Returns (accepted issues, dropped-issue notes, whether the check actually ran)."""
    blocks = _spelling_source_blocks(profile)
    if not blocks:
        return [], [], True

    try:
        llm_response = _call_llm_spelling_check(blocks, client=client)
    except CvImproverError:
        return [], [], False

    accepted: list[SpellingIssue] = []
    dropped: list[str] = []
    for item in llm_response.issues:
        if not (0 <= item.index < len(blocks)):
            continue
        label, original = blocks[item.index]
        corrected = item.corrected.strip()
        if not corrected:
            continue
        if _normalise_comparison_text(corrected) == _normalise_comparison_text(original):
            continue
        if _numbers_in(corrected) != _numbers_in(original):
            dropped.append(f"{label}: suggested fix changed a number — kept original wording")
            continue
        original_skills = match_skill_canonicals(original, ontology)
        corrected_skills = match_skill_canonicals(corrected, ontology)
        if corrected_skills != original_skills:
            dropped.append(f"{label}: suggested fix changed a skill mention — kept original")
            continue
        if _is_disguised_rewrite(original, corrected):
            dropped.append(f"{label}: suggested fix looked like a rewrite, not a spelling fix")
            continue
        accepted.append(SpellingIssue(source_label=label, original=original, corrected=corrected))
    return accepted, dropped, True


def _has_measurable_result(text: str) -> bool:
    return bool(_METRIC_PATTERN.search(text))


def _is_substantive_experience_evidence(text: str, entry_title: str | None) -> bool:
    """Do not score a title/company fallback as though it were an experience bullet."""
    compact = " ".join(text.split())
    if len(compact.split()) < 3:
        return False
    title = (entry_title or "").casefold()
    return compact.casefold() != title and bool(_DESCRIPTIVE_ACTION_PATTERN.search(compact))


def _impact_text_for(entry: object) -> str | None:
    impact_evidence = getattr(entry, "impact_evidence", None)
    if impact_evidence and impact_evidence.strip():
        return impact_evidence.strip()
    evidence = getattr(entry, "evidence", "").strip()
    if _is_substantive_experience_evidence(evidence, getattr(entry, "title", None)):
        return evidence
    return None


def _experience_title_positions(profile: CvProfile, document_text: str) -> list[int | None]:
    """Find each extracted title in document order, preferring its nearby employer."""
    positions: list[int | None] = []
    cursor = 0
    for entry in profile.experience_entries:
        title = (entry.title or "").strip()
        if not title:
            positions.append(None)
            continue
        title_positions = [
            match.start() for match in re.finditer(re.escape(title), document_text, re.IGNORECASE)
        ]
        after_cursor = [position for position in title_positions if position >= cursor]
        candidates = after_cursor or title_positions
        if not candidates:
            positions.append(None)
            continue
        company = (entry.employer or "").strip()
        company_match = (
            re.search(re.escape(company), document_text, re.IGNORECASE) if company else None
        )
        if company_match is not None:
            start = min(candidates, key=lambda position: abs(company_match.start() - position))
        else:
            start = candidates[0]
        positions.append(start)
        cursor = start + len(title)
    return positions


def _line_containing(role_text: str, match: re.Match[str]) -> str | None:
    start = role_text.rfind("\n", 0, match.start()) + 1
    end = role_text.find("\n", match.end())
    line = role_text[start : len(role_text) if end == -1 else end].strip()
    return line or None


def _metric_line(role_text: str) -> str | None:
    """Return the literal line containing a measurable result, never a generated summary."""
    match = _METRIC_PATTERN.search(role_text)
    return _line_containing(role_text, match) if match is not None else None


def _descriptive_line(role_text: str) -> str | None:
    """Return a real responsibility line when a role has no visible metric."""
    for line in role_text.splitlines():
        compact = line.strip()
        if _DESCRIPTIVE_ACTION_PATTERN.search(compact):
            return compact
    return None


def enrich_experience_impact_evidence(profile: CvProfile, document_text: str) -> CvProfile:
    """Capture a short literal metric-bearing quote for each known role before redaction.

    The extractor may correctly identify a role while returning only its title/company
    as evidence. This helper looks from that literal title only until the next known
    role, so a result from another job is never assigned to it.
    """
    if not document_text or not profile.experience_entries:
        return profile
    positions = _experience_title_positions(profile, document_text)
    updated = []
    for index, entry in enumerate(profile.experience_entries):
        start = positions[index]
        if start is None:
            updated.append(entry)
            continue
        following_starts = [position for position in positions[index + 1 :] if position is not None]
        end = next(
            (position for position in following_starts if position > start), len(document_text)
        )
        role_text = document_text[start:end]
        impact_evidence = (
            _metric_line(role_text) or _descriptive_line(role_text) or entry.impact_evidence
        )
        updated.append(entry.model_copy(update={"impact_evidence": impact_evidence}))
    return profile.model_copy(update={"experience_entries": updated})


def _find_unquantified_bullets(profile: CvProfile) -> list[UnquantifiedBullet]:
    unquantified: list[UnquantifiedBullet] = []
    for entry in profile.experience_entries:
        text = _impact_text_for(entry)
        if not text or _has_measurable_result(text):
            continue
        label = f"{entry.title or 'Untitled role'} @ {entry.employer or 'unknown employer'}"
        unquantified.append(UnquantifiedBullet(source_label=label, text=text))
    return unquantified


def _spelling_check(issues: list[SpellingIssue], *, checked: bool) -> QualityCheck:
    max_score = 15.0
    if not checked:
        return QualityCheck(
            id="spelling_grammar",
            label="Spelling & Grammar",
            status="not_checked",
            score=0.0,
            max_score=max_score,
            message="Spelling/grammar check could not run (the LLM call failed).",
            suggestion="Try generating the report again in a moment.",
        )
    count = len(issues)
    if count == 0:
        return QualityCheck(
            id="spelling_grammar",
            label="Spelling & Grammar",
            status="pass",
            score=max_score,
            max_score=max_score,
            message="No spelling or grammar issues detected.",
        )
    score = max(0.0, max_score - count * 3.0)
    status: QualityCheckStatus = "warning" if count <= 3 else "fail"
    return QualityCheck(
        id="spelling_grammar",
        label="Spelling & Grammar",
        status=status,
        score=score,
        max_score=max_score,
        message=f"{count} spelling/grammar issue(s) detected.",
        suggestion="Review the flagged phrases below and apply the suggested corrections.",
    )


def _impact_check(profile: CvProfile, unquantified: list[UnquantifiedBullet]) -> QualityCheck:
    max_score = 25.0
    total = sum(1 for entry in profile.experience_entries if _impact_text_for(entry))
    if total == 0:
        return QualityCheck(
            id="impact_quantification",
            label="Impact Quantification",
            status="fail",
            score=0.0,
            max_score=max_score,
            message="No experience entries with descriptive text were found to assess.",
            suggestion=(
                "Add measurable outcomes (%, $, counts, time saved) to your experience bullets."
            ),
        )
    quantified = total - len(unquantified)
    fraction = quantified / total
    score = round(max_score * fraction, 1)
    if fraction >= 0.7:
        status: QualityCheckStatus = "pass"
    elif fraction >= 0.3:
        status = "warning"
    else:
        status = "fail"
    return QualityCheck(
        id="impact_quantification",
        label="Impact Quantification",
        status=status,
        score=score,
        max_score=max_score,
        message=f"{quantified} of {total} experience entries include a measurable result.",
        suggestion=(
            "Add a number, percentage, or metric to the bullets flagged below."
            if unquantified
            else None
        ),
    )


_SUMMARY_LABEL_PATTERN = re.compile(r"summary|profile|objective|about", re.IGNORECASE)


def _summary_presence(profile: CvProfile) -> bool | None:
    """True/False if the deterministic parser's section labels can confirm it,
    None if this profile has no section data to check (the LLM extraction path
    never populates `sections` — see `cv/llm_extraction.py`) — reported as
    "not verifiable", never silently asserted as missing."""
    if not profile.sections:
        return None
    return any(
        _SUMMARY_LABEL_PATTERN.search(section.normalized_label or section.label)
        for section in profile.sections
    )


def _sections_check(profile: CvProfile) -> QualityCheck:
    max_score = 15.0
    has_skills = bool(profile.skills)
    summary_present = _summary_presence(profile)

    score = 10.0 if has_skills else 0.0
    if summary_present is not False:
        score += 5.0  # unknown is not penalized — see _summary_presence docstring

    missing = []
    if not has_skills:
        missing.append("skills")
    if summary_present is False:
        missing.append("summary")
    summary_note = (
        "a professional summary section was detected"
        if summary_present
        else "no professional summary section was detected"
        if summary_present is False
        else "professional summary presence could not be verified for this extraction method"
    )
    if not missing:
        status: QualityCheckStatus = "pass"
    elif len(missing) == 1:
        status = "warning"
    else:
        status = "fail"
    skills_state = "present" if has_skills else "missing"
    return QualityCheck(
        id="missing_sections",
        label="Missing Sections",
        status=status,
        score=round(score, 1),
        max_score=max_score,
        message=f"Skills section: {skills_state}. {summary_note.capitalize()}.",
        suggestion=(
            f"Add a {' and '.join(missing)} section to strengthen your CV structure."
            if missing
            else None
        ),
    )


def _contact_check(
    candidate_name: str | None,
    contact_info: str | None,
    contact_signals: ContactSignals,
) -> QualityCheck:
    """A channel counts as present if it's in the CV's own text OR the optional form
    field — never flagged missing just because the optional field was left blank."""
    max_score = 10.0
    has_name = bool((candidate_name or "").strip()) or contact_signals.has_name
    form_signals = detect_contact_signals(contact_info or "")
    has_email = contact_signals.has_email or form_signals.has_email
    has_phone = contact_signals.has_phone or form_signals.has_phone
    has_linkedin = contact_signals.has_linkedin_or_github or form_signals.has_linkedin_or_github

    channels = {"email": has_email, "phone": has_phone, "LinkedIn/GitHub": has_linkedin}
    missing_channels = [label for label, present in channels.items() if not present]
    found_count = sum(channels.values()) + (1 if has_name else 0)
    score = round(max_score * found_count / 4, 1)

    if not missing_channels and has_name:
        return QualityCheck(
            id="contact_info",
            label="Contact Information",
            status="pass",
            score=max_score,
            max_score=max_score,
            message="Name and contact details (email, phone, LinkedIn/GitHub) are all present.",
        )
    if not missing_channels:
        return QualityCheck(
            id="contact_info",
            label="Contact Information",
            status="warning",
            score=score,
            max_score=max_score,
            message=(
                "Email, phone, and LinkedIn/GitHub were found in your CV. No name was "
                "provided for the document header."
            ),
            suggestion="Add your name using the optional field before generating your CV.",
        )
    missing_text = " and ".join(missing_channels) if len(missing_channels) <= 2 else (
        ", ".join(missing_channels[:-1]) + f", and {missing_channels[-1]}"
    )
    status: QualityCheckStatus = "fail" if found_count == 0 else "warning"
    return QualityCheck(
        id="contact_info",
        label="Contact Information",
        status=status,
        score=score,
        max_score=max_score,
        message=f"Couldn't find {missing_text} in your CV or the optional fields.",
        suggestion=(
            f"Add your {missing_text} — either make sure it's visible in your CV or fill it "
            "into the optional fields before generating your CV. This is never stored — "
            "it's used only to fill in that one document."
        ),
    )


def _ats_check(profile: CvProfile) -> QualityCheck:
    max_score = 15.0
    has_education = profile.education_detail is not None
    has_experience = bool(profile.experience_entries)
    has_skills = bool(profile.skills)
    high_confidence = profile.extraction_quality.extraction_confidence_label != "low"
    present_count = sum([has_education, has_experience, has_skills, high_confidence])

    missing = []
    if not has_education:
        missing.append("education")
    if not has_experience:
        missing.append("experience")
    if not has_skills:
        missing.append("skills")
    if not high_confidence:
        missing.append("high-confidence extraction")

    score = round(max_score * present_count / 4, 1)
    status: QualityCheckStatus = (
        "pass" if present_count == 4 else "warning" if present_count >= 2 else "fail"
    )
    message = (
        "All standard sections were parsed cleanly."
        if not missing
        else f"Missing or low-confidence: {', '.join(missing)}."
    )
    return QualityCheck(
        id="ats_readiness",
        label="ATS Readiness",
        status=status,
        score=score,
        max_score=max_score,
        message=message,
        suggestion=(
            "Use standard section headers (Experience, Education, Skills) and simple, "
            "single-column formatting so ATS parsers can read your CV correctly."
            if missing
            else None
        ),
    )


def _skill_gap_check(recommended: list[RecommendedSkill]) -> QualityCheck:
    max_score = 20.0
    top = recommended[:5]
    score = max(0.0, max_score - len(top) * 4.0)
    status: QualityCheckStatus = "pass" if len(top) <= 1 else "warning" if len(top) <= 3 else "fail"
    if not recommended:
        message = "No significant skill gaps found against your matched jobs."
    else:
        names = ", ".join(item.canonical_skill for item in top)
        message = f"{len(recommended)} skill gap(s) found vs. your matched jobs, e.g. {names}."
    return QualityCheck(
        id="market_skill_gap",
        label="Market Skill Gap",
        status=status,
        score=round(score, 1),
        max_score=max_score,
        message=message,
        suggestion=(
            "See the recommended-skills list below for what the market is actually asking for."
            if recommended
            else None
        ),
    )


def generate_cv_quality_report(
    profile: CvProfile,
    skill_gap: Sequence[SkillGapEntry],
    matched_jobs: Sequence[RecommendationOutput],
    *,
    candidate_name: str | None = None,
    contact_info: str | None = None,
    contact_signals: ContactSignals | None = None,
    ontology: SkillsOntology | None = None,
    client: OpenAI | None = None,
) -> CvQualityReport:
    """Score and diagnose a CV. Never raises for a spelling-check LLM failure — that
    one category is reported as `not_checked` and excluded from the score instead,
    since the other five checks are fully deterministic and don't need to fail with it.
    """
    actual_ontology = ontology or load_ontology()
    actual_signals = contact_signals or ContactSignals()

    spelling_issues, dropped_spelling, spelling_checked = _detect_spelling_issues(
        profile, actual_ontology, client=client
    )
    unquantified = _find_unquantified_bullets(profile)
    recommended = _recommended_skills(skill_gap, len(matched_jobs))

    checks = [
        _spelling_check(spelling_issues, checked=spelling_checked),
        _impact_check(profile, unquantified),
        _sections_check(profile),
        _contact_check(candidate_name, contact_info, actual_signals),
        _ats_check(profile),
        _skill_gap_check(recommended),
    ]

    scored_checks = [check for check in checks if check.status != "not_checked"]
    total_score = sum(check.score for check in scored_checks)
    total_max = sum(check.max_score for check in scored_checks) or 1.0
    overall_score = round(100 * total_score / total_max)

    is_degraded = profile.extraction_quality.extraction_confidence_label == "low"
    degraded_reason = None
    if is_degraded:
        degraded_reason = profile.extraction_quality.abstention_reason or (
            "Source CV extraction confidence was low; this report may be incomplete."
        )

    return CvQualityReport(
        overall_score=overall_score,
        checks=checks,
        spelling_issues=spelling_issues,
        unquantified_bullets=unquantified,
        recommended_skills_to_develop=recommended,
        dropped_spelling_suggestions=dropped_spelling,
        source_degraded=is_degraded,
        source_degraded_reason=degraded_reason,
    )


__all__ = [
    "ContactSignals",
    "CvQualityReport",
    "QualityCheck",
    "SpellingIssue",
    "UnquantifiedBullet",
    "detect_contact_signals",
    "enrich_experience_impact_evidence",
    "generate_cv_quality_report",
]
