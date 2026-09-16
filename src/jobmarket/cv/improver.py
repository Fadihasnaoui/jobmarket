"""Data-driven CV improvement: reword real facts, never invent new ones.

This module does two things, kept deliberately separate so they can never blur:

1. IMPROVE what's already in the CV — reword each real, already-grounded experience
   entry and produce a professional summary, using market context drawn from the
   candidate's own matched jobs. One LLM call does the rewording.
2. RECOMMEND what's missing — pulled straight from the already-computed skill-gap
   input, annotated with real demand. These skills are never added to the CV body.

Grounding discipline (same spirit as `cv/llm_guard.py`, applied to free text instead
of structured skill candidates): the LLM's job is to reword existing facts, not to
add new ones. Since it would be unsafe to trust that on the model's word alone, every
piece of LLM-generated text is validated after the fact and rejected — falling back
to the original, already-grounded wording — if it:
  - mentions any ontology-resolvable skill the candidate's grounded profile doesn't
    have (checked via `skills.matcher.match_skill_canonicals`, the same deterministic
    skill-matching machinery used everywhere else in this pipeline);
  - introduces a number (a metric, a percentage, a headcount, ...) that wasn't in the
    original evidence text for that entry;
  - mentions a *different* employer than the one that entry is actually describing.

Company/title/dates are never even given a chance to drift: the LLM is asked only
for an `index` and an `improved_description` per experience, and every other field
on `ImprovedExperienceEntry` is copied verbatim from the original, already-grounded
`ExperienceEntry` — there is no "generate then compare" step for those fields because
there is no path for the LLM to touch them at all.

No deterministic fallback exists for "improve this CV" the way there is for
extraction — rewording is inherently a generative task. If the LLM call fails
outright (bad key, network error, malformed JSON after one retry), this module
raises `CvImproverError` rather than fabricating a non-LLM "improved" CV; the caller
must surface that honestly instead of pretending the feature ran.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence

import structlog
from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from jobmarket.config import get_settings
from jobmarket.cv.profile import CvProfile, ExperienceEntry
from jobmarket.cv.workflow import RecommendationOutput
from jobmarket.skills.matcher import match_skill_canonicals
from jobmarket.skills.ontology import SkillsOntology, load_ontology

logger = structlog.get_logger(__name__)

CV_IMPROVER_VERSION = "cv-improver-v1"
MAX_ATTEMPTS = 2  # one retry, with a simplified/stricter prompt, before raising
MAX_COMPLETION_TOKENS = 3000  # a summary + a handful of reworded bullets, not full extraction

# Trailing "." only joins the number if followed by more digits (a real decimal),
# never a sentence-ending period — and unlike a comma (a thousands separator, safe
# to normalize away), a decimal point changes the actual value and must never be
# stripped: "66" and "6.6" have to stay distinguishable, or a reworded "$4.55K"
# would wrongly read as unchanged from an original "$455K".
_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?")


class CvImproverError(RuntimeError):
    """Raised when LLM-based CV improvement cannot produce a usable result.

    There is no deterministic fallback for this feature (unlike CV extraction) —
    callers must surface this honestly rather than inventing a non-LLM substitute.
    """


class SkillGapEntry(BaseModel):
    """One missing skill for a CV, ranked by demand.

    Mirrors `jobmarket.api.models.SkillGapItem`'s fields exactly, but is defined here
    instead of imported from it: `cv/` must not depend on `api/` (the API layer
    depends on `cv/`, never the reverse). Callers in the API layer construct this
    from their own `SkillGapItem` list (e.g. `SkillGapEntry(**item.model_dump())`).
    """

    model_config = ConfigDict(frozen=True)

    canonical_skill: str
    category: str | None = None
    missing_in_top_matches: int
    overall_job_demand: int


class RecommendedSkill(BaseModel):
    """A missing skill surfaced separately from the CV body, with real demand."""

    model_config = ConfigDict(frozen=True)

    canonical_skill: str
    category: str | None = None
    demand_pct_of_matches: float
    overall_job_demand: int


class ImprovedExperienceEntry(BaseModel):
    """One experience entry with reworded phrasing; every fact field is copied
    verbatim from the original `ExperienceEntry` — never LLM-sourced."""

    model_config = ConfigDict(frozen=True)

    title: str | None
    employer: str | None
    start_date: str | None
    end_date: str | None
    entry_type: str
    original_evidence: str
    improved_description: str


class ImprovedCvResult(BaseModel):
    """Result of `generate_improved_cv`."""

    model_config = ConfigDict(frozen=True)

    improved_summary: str
    improved_experiences: list[ImprovedExperienceEntry] = Field(default_factory=list)
    skills_section: list[str] = Field(default_factory=list)
    recommended_skills_to_develop: list[RecommendedSkill] = Field(default_factory=list)
    changes_explanation: list[str] = Field(default_factory=list)
    source_degraded: bool = False
    source_degraded_reason: str | None = None
    dropped_ungrounded_mentions: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You improve the wording of a candidate's CV for a job-matching platform. You "
    "ONLY reword and restructure facts that are already stated — you never invent "
    "new skills, achievements, metrics, employers, titles, or dates. Every rewritten "
    "bullet must describe the exact same real fact as the original evidence, just "
    "with clearer, more market-aligned professional language. Rewording should add "
    "clarity and impact, not remove real content — if a bullet already names real "
    "work (even briefly), keep that substance; do not collapse it down to a single "
    "generic word.\n"
    "The user message gives you an ALLOWED SKILL VOCABULARY list. You may name a "
    "skill ONLY if it appears in that exact list, verbatim, OR if it is already "
    "part of THAT SPECIFIC job's own real title or evidence text given to you (e.g. "
    "if that job's title is \"Data Science Intern\", describing it as data-science "
    "work is restating a real fact, not inventing one — that's allowed). What is "
    "NEVER allowed: naming a skill or domain word that belongs to a DIFFERENT job, "
    "or that appears nowhere in the candidate's real text at all (e.g. do not call "
    "an \"Observation Intern\" role \"data analysis\" work just because a different "
    "job on this CV happens to be data-related).\n"
    "If you are unsure whether a phrase would add a new fact or an unlisted skill, "
    "keep the original wording for that part instead of guessing — an unchanged "
    "bullet is always safer than a fabricated or over-thinned one. Respond with a "
    "single strict JSON object matching the given schema exactly — no prose, no "
    "markdown fences."
)

_SIMPLIFIED_SYSTEM_PROMPT = (
    "Output ONLY one JSON object matching the schema below. No prose, no markdown, "
    "no explanations. Reword facts, never add new ones. Only name a skill if it is "
    "in the ALLOWED SKILL VOCABULARY list given — never a domain/field name like "
    "\"Data Science\" unless that exact phrase is in that list. If unsure, copy the "
    "original wording for that field."
)


class _LlmExperienceRewrite(BaseModel):
    model_config = {"extra": "ignore"}
    index: int
    improved_description: str = ""


class _LlmImproverResponse(BaseModel):
    model_config = {"extra": "ignore"}
    professional_summary: str = ""
    experiences: list[_LlmExperienceRewrite] = Field(default_factory=list)


def _schema_description(grounded_skills: list[str]) -> str:
    vocab = ", ".join(grounded_skills) if grounded_skills else "(none extracted)"
    return (
        "Return a JSON object with exactly these fields:\n"
        '- "professional_summary": string, 2-4 sentences, built only from the real '
        "profile facts given above (career level, total experience, grounded skills). "
        "Do not name a skill or domain/field word unless it is in the ALLOWED SKILL "
        "VOCABULARY list or already part of one of the candidate's real job titles.\n"
        '- "experiences": array of {"index": integer (copy the index given for that '
        'job below, exactly), "improved_description": string — a reworded version of '
        "that job's evidence text: same facts, clearer/stronger professional phrasing, "
        "preserving the real substance already there (don't thin it down). Name a "
        "skill only from the allowed vocabulary or that job's own title/evidence. One "
        "entry per job given below, same indices, no extra jobs invented.\n\n"
        f"REMINDER — ALLOWED SKILL VOCABULARY (the ONLY skills you may name, verbatim): "
        f"{vocab}"
    )


def _build_user_prompt(profile: CvProfile, matched_jobs: Sequence[RecommendationOutput]) -> str:
    lines: list[str] = []
    grounded_skills = sorted(_grounded_skill_names(profile))
    lines.append(f"Candidate career level: {profile.candidate.career_level}")
    if profile.candidate.experience.total_years is not None:
        lines.append(f"Total years of experience: {profile.candidate.experience.total_years}")
    lines.append(
        "ALLOWED SKILL VOCABULARY — the skills you may name, verbatim, anywhere in "
        "your output (exactly these strings, not synonyms): "
        + (", ".join(grounded_skills) if grounded_skills else "(none extracted)")
    )
    lines.append(
        "Domain/field labels for your own context — do not write these into the CV "
        "as if they were a claimed skill UNLESS the same word is also literally part "
        "of the specific job's own title/evidence you are rewording (see rule in the "
        "system prompt): "
        + (", ".join(sorted({d.value for d in profile.inferred_domains})) or "(none)")
    )
    lines.append("")
    lines.append("Real work experience (reword each entry's evidence text; keep the facts):")
    if not profile.experience_entries:
        lines.append("(No experience entries were extracted from this CV.)")
    for index, entry in enumerate(profile.experience_entries):
        lines.append(
            f"[{index}] title={entry.title or 'unknown'} | employer={entry.employer or 'unknown'} "
            f"| dates={entry.start_date or '?'}-{entry.end_date or '?'}\n"
            f"    evidence: {entry.evidence}"
        )
    lines.append("")
    lines.append(_build_market_context(matched_jobs))
    lines.append("")
    lines.append(_schema_description(grounded_skills))
    return "\n".join(lines)


def _build_market_context(matched_jobs: Sequence[RecommendationOutput]) -> str:
    if not matched_jobs:
        return "No matched job postings were available for market context."
    titles = [job.title for job in matched_jobs[:5]]
    skill_counter: Counter[str] = Counter()
    for job in matched_jobs:
        skill_counter.update(job.matched_skills)
    top_skills = [skill for skill, _ in skill_counter.most_common(8)]
    context = (
        "Real job postings this candidate matched with include titles like: "
        + "; ".join(titles)
        + ". Align your phrasing/vocabulary with how such postings describe similar "
        "work (but only for skills the candidate actually has, listed above)."
    )
    if top_skills:
        context += " Skills most often shared with these postings: " + ", ".join(top_skills) + "."
    return context


def _build_messages(
    profile: CvProfile, matched_jobs: Sequence[RecommendationOutput], *, simplified: bool
) -> list[dict[str, str]]:
    system = _SIMPLIFIED_SYSTEM_PROMPT if simplified else _SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_prompt(profile, matched_jobs)},
    ]


def _client() -> OpenAI:
    settings = get_settings()
    if not settings.llm_api_key:
        raise CvImproverError("LLM_API_KEY is not set (see .env.example)")
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def call_llm_cv_improvement(
    profile: CvProfile,
    matched_jobs: Sequence[RecommendationOutput],
    *,
    client: OpenAI | None = None,
) -> _LlmImproverResponse:
    """One LLM call producing strict-JSON reworded CV text. Retries once on failure."""
    settings = get_settings()
    llm_client = client or _client()
    last_error = "unknown error"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        messages = _build_messages(profile, matched_jobs, simplified=attempt > 1)
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
            raise CvImproverError(last_error) from exc
        except Exception as exc:  # network/timeout/rate-limit/auth/sdk errors
            raise CvImproverError(f"LLM call failed: {exc}") from exc
        raw = response.choices[0].message.content or ""
        try:
            return _LlmImproverResponse.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = f"invalid JSON: {exc}"
            if attempt < MAX_ATTEMPTS:
                continue
            raise CvImproverError(
                f"LLM did not return valid JSON after {MAX_ATTEMPTS} attempts: {last_error}"
            ) from exc
    raise CvImproverError(f"LLM improvement failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _log_info(event: str, **values: object) -> None:
    try:
        logger.info(event, **values)
    except ValueError:
        # Test harnesses may close captured stdout after Typer configures structlog.
        return


def _grounded_skill_names(profile: CvProfile) -> set[str]:
    return {skill.canonical_skill for skill in profile.skills}


def _numbers_in(text: str) -> set[str]:
    return {token.replace(",", "") for token in _NUMBER_PATTERN.findall(text)}


def _other_employer_names(profile: CvProfile, exclude_employer: str | None) -> list[str]:
    seen: list[str] = []
    for entry in profile.experience_entries:
        if entry.employer and entry.employer != exclude_employer and entry.employer not in seen:
            seen.append(entry.employer)
    return seen


def _mentions_other_employer(text: str, other_employers: list[str]) -> str | None:
    lowered = text.casefold()
    for name in other_employers:
        if name.casefold() in lowered:
            return name
    return None


def _own_grounded_terms(*texts: str | None, ontology: SkillsOntology) -> set[str]:
    """Skill/domain-shaped terms already true for a specific piece of real CV text.

    Restating a job's own real title (e.g. mentioning "Data Science" when that job's
    actual title is "Data Science Intern") isn't a new fact — it's already grounded
    for that entry, even if "Data Science" never made it into the candidate's formal
    skills list. This is deliberately narrower than the global skills list: it's
    computed per-entry, from that entry's own title/evidence only, so it can't be
    used to smuggle in a domain word that belongs to a *different* job or that isn't
    written anywhere in the CV at all — see `_safe_experience_description`/
    `_safe_summary`, which are the only callers.
    """
    combined = " ".join(text for text in texts if text)
    return match_skill_canonicals(combined, ontology)


def _safe_experience_description(
    original: ExperienceEntry,
    candidate_text: str | None,
    *,
    allowed_skills: set[str],
    other_employers: list[str],
    ontology: SkillsOntology,
    dropped: list[str],
) -> str:
    label = original.employer or original.title or "unknown job"
    if not candidate_text or not candidate_text.strip():
        return original.evidence

    own_terms = _own_grounded_terms(original.title, original.evidence, ontology=ontology)
    mentioned = match_skill_canonicals(candidate_text, ontology)
    extra_skills = mentioned - allowed_skills - own_terms
    if extra_skills:
        dropped.append(
            f"experience[{label}]: ungrounded skill mention(s) {sorted(extra_skills)} — "
            "kept original wording"
        )
        return original.evidence

    if not _numbers_in(candidate_text) <= _numbers_in(original.evidence):
        dropped.append(
            f"experience[{label}]: introduced a number not present in the original — "
            "kept original wording"
        )
        return original.evidence

    other = _mentions_other_employer(candidate_text, other_employers)
    if other is not None:
        dropped.append(
            f"experience[{label}]: mentioned a different employer ({other}) — "
            "kept original wording"
        )
        return original.evidence

    return candidate_text.strip()


def _fallback_summary(profile: CvProfile) -> str:
    level = profile.candidate.career_level
    level_text = "Experienced" if level == "unknown" else level.replace("_", " ").title()
    skills = sorted(_grounded_skill_names(profile))[:5]
    if skills:
        return f"{level_text} professional with grounded experience in {', '.join(skills)}."
    return f"{level_text} professional profile extracted from the submitted CV."


def _safe_summary(
    candidate_text: str,
    profile: CvProfile,
    *,
    allowed_skills: set[str],
    ontology: SkillsOntology,
    dropped: list[str],
) -> str:
    if not candidate_text or not candidate_text.strip():
        return _fallback_summary(profile)

    own_terms: set[str] = set()
    for entry in profile.experience_entries:
        own_terms |= _own_grounded_terms(entry.title, entry.evidence, ontology=ontology)
    allowed_skills = allowed_skills | own_terms

    mentioned = match_skill_canonicals(candidate_text, ontology)
    extra_skills = mentioned - allowed_skills
    if extra_skills:
        dropped.append(
            f"summary: ungrounded skill mention(s) {sorted(extra_skills)} — used fallback summary"
        )
        return _fallback_summary(profile)

    return candidate_text.strip()


def _recommended_skills(
    skill_gap: Sequence[SkillGapEntry], matched_job_count: int
) -> list[RecommendedSkill]:
    if matched_job_count <= 0:
        return [
            RecommendedSkill(
                canonical_skill=item.canonical_skill,
                category=item.category,
                demand_pct_of_matches=0.0,
                overall_job_demand=item.overall_job_demand,
            )
            for item in skill_gap
        ]
    return [
        RecommendedSkill(
            canonical_skill=item.canonical_skill,
            category=item.category,
            demand_pct_of_matches=round(item.missing_in_top_matches / matched_job_count * 100, 1),
            overall_job_demand=item.overall_job_demand,
        )
        for item in skill_gap
    ]


def _build_changes_explanation(
    profile: CvProfile,
    improved_experiences: list[ImprovedExperienceEntry],
    recommended: list[RecommendedSkill],
    dropped: list[str],
) -> list[str]:
    lines: list[str] = []
    total = len(profile.experience_entries)
    if total:
        reworded = sum(
            1
            for imp in improved_experiences
            if imp.improved_description.strip() != imp.original_evidence.strip()
        )
        lines.append(
            f"Reworded {reworded} of {total} experience bullet(s) with clearer, "
            "market-aligned phrasing; employer, title, and dates are unchanged."
        )
    lines.append(
        "Generated a professional summary from your grounded skills and career level — "
        "no skill was added that wasn't already in your extracted profile."
    )
    if recommended:
        lines.append(
            f"Listed {len(recommended)} skill(s) worth developing based on real demand "
            "across your matched jobs, in a separate section — these are NOT added to "
            "your skills list."
        )
    if dropped:
        lines.append(
            f"{len(dropped)} generated phrase(s) failed the grounding check and were "
            "replaced with your original wording (see dropped_ungrounded_mentions)."
        )
    return lines


def generate_improved_cv(
    profile: CvProfile,
    skill_gap: Sequence[SkillGapEntry],
    matched_jobs: Sequence[RecommendationOutput],
    *,
    ontology: SkillsOntology | None = None,
    client: OpenAI | None = None,
) -> ImprovedCvResult:
    """Generate a market-aligned, grounded CV improvement.

    Assumes `skill_gap` and `matched_jobs` were computed over the same top-N matches
    (so `demand_pct_of_matches` — "appears in N% of jobs matching your profile" — is a
    coherent percentage); mismatched inputs won't error, but the percentage will be
    meaningless. Raises `CvImproverError` if the LLM call fails outright — there is no
    deterministic fallback for this feature (see module docstring).

    `source_degraded` mirrors this profile's own `extraction_quality.
    extraction_confidence_label == "low"` — the same signal already used elsewhere in
    this pipeline to flag a shaky extraction, not a new one invented for this feature.
    """
    actual_ontology = ontology or load_ontology()
    allowed_skills = _grounded_skill_names(profile)
    dropped: list[str] = []

    llm_response = call_llm_cv_improvement(profile, matched_jobs, client=client)

    summary = _safe_summary(
        llm_response.professional_summary,
        profile,
        allowed_skills=allowed_skills,
        ontology=actual_ontology,
        dropped=dropped,
    )

    rewrites_by_index = {item.index: item.improved_description for item in llm_response.experiences}
    improved_experiences: list[ImprovedExperienceEntry] = []
    for index, original in enumerate(profile.experience_entries):
        other_employers = _other_employer_names(profile, original.employer)
        description = _safe_experience_description(
            original,
            rewrites_by_index.get(index),
            allowed_skills=allowed_skills,
            other_employers=other_employers,
            ontology=actual_ontology,
            dropped=dropped,
        )
        improved_experiences.append(
            ImprovedExperienceEntry(
                title=original.title,
                employer=original.employer,
                start_date=original.start_date,
                end_date=original.end_date,
                entry_type=original.entry_type,
                original_evidence=original.evidence,
                improved_description=description,
            )
        )

    recommended = _recommended_skills(skill_gap, len(matched_jobs))
    changes = _build_changes_explanation(profile, improved_experiences, recommended, dropped)

    is_degraded = profile.extraction_quality.extraction_confidence_label == "low"
    degraded_reason = None
    if is_degraded:
        degraded_reason = profile.extraction_quality.abstention_reason or (
            "Source CV extraction confidence was low; some experience or skill "
            "details may be missing or incomplete."
        )

    if dropped:
        _log_info("cv_improver_dropped_ungrounded_text", count=len(dropped), details=dropped)

    return ImprovedCvResult(
        improved_summary=summary,
        improved_experiences=improved_experiences,
        skills_section=sorted(allowed_skills),
        recommended_skills_to_develop=recommended,
        changes_explanation=changes,
        source_degraded=is_degraded,
        source_degraded_reason=degraded_reason,
        dropped_ungrounded_mentions=dropped,
    )


__all__ = [
    "CV_IMPROVER_VERSION",
    "CvImproverError",
    "ImprovedCvResult",
    "ImprovedExperienceEntry",
    "RecommendedSkill",
    "SkillGapEntry",
    "call_llm_cv_improvement",
    "generate_improved_cv",
]
