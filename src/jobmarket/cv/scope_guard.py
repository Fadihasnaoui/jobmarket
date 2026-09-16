"""Out-of-scope detection for profiles with zero grounded tech skills.

`cv/matching.py` (untouched) already suppresses individual tech-job rows for a
business-domain candidate with no technical evidence (see
`_business_profile_technical_role_mismatch`). But when a profile has *no* domain
signal recognized as business either (or the LLM/deterministic path simply produced
weak tagging), matching can still surface a handful of low-signal tech rows whose own
explanation reads "the CV and job share no explicit skills" — a bad result presented as
a normal one.

This module adds one narrow, explicit guard upstream of matching: only when there is
zero grounded tech-skill evidence AND the detected domains/roles are recognizably
non-tech does it short-circuit with an honest message instead of 10 near-random rows.
It must never fire for a genuine tech profile that merely has few skills (e.g. a junior
developer with 2 skills) — that CV still has technical domain/role signal, which is
exactly what keeps this guard from tripping.
"""

from __future__ import annotations

from jobmarket.cv.matching import BUSINESS_PROFILE_DOMAINS, TECHNICAL_RECOMMENDATION_DOMAINS
from jobmarket.cv.profile import CvProfile
from jobmarket.skills.ontology import get_ontology

OUT_OF_SCOPE_MESSAGE = (
    "This profile appears to fall outside the IT/tech scope this platform covers; "
    "recommendations may not be relevant."
)

# Non-tech role families the extractors (LLM or deterministic) may tag; deliberately a
# curated subset — "Project Management"/"Operations" are excluded since they're common
# in tech orgs too and shouldn't, by themselves, prove a profile is non-tech.
_NON_TECH_ROLE_FAMILIES = {
    "Sales",
    "Customer Service",
    "Management",
    "Marketing",
    "Business Development",
    "Human Resources",
    "Finance",
    "Insurance",
    "Customer Relationship Management",
    "Administrative Operations",
}


def _has_tech_skill_evidence(profile: CvProfile) -> bool:
    """Any grounded skill outside the ontology's generic `soft_skill` category counts.

    The ontology's `soft_skill` category (Negotiation, Client Relationship Management,
    Vendor Management, Public Speaking, ...) is common to sales/management/support
    roles too, so a match there alone must not prove a profile is technical — every
    other category (language, framework, tool, platform, database, cloud, methodology,
    and the tech-heavy `domain` category) does.
    """
    ontology = get_ontology()
    for skill in profile.skills:
        entry = ontology.get(skill.canonical_skill)
        if entry is not None and entry.category != "soft_skill":
            return True
    return False


def out_of_scope_reason(profile: CvProfile) -> str | None:
    """Return an honest out-of-scope message, or None when matching should proceed.

    Fires only on the combination of zero grounded *tech* skills AND domain/role
    evidence that is recognizably non-tech — never on domain/role silence alone (that
    case is already covered by the low-confidence warning, not this guard) and never
    when any technical skill or domain is present (that keeps thin tech profiles
    unaffected, even a junior developer CV with only 2 skills).
    """
    if _has_tech_skill_evidence(profile):
        return None
    candidate_domains = set(profile.candidate.preferred_domains)
    if candidate_domains & TECHNICAL_RECOMMENDATION_DOMAINS:
        return None
    role_values = {role.value for role in profile.inferred_roles}
    non_tech_signal = (candidate_domains & BUSINESS_PROFILE_DOMAINS) or (
        role_values & _NON_TECH_ROLE_FAMILIES
    )
    if not non_tech_signal:
        return None
    return OUT_OF_SCOPE_MESSAGE


__all__ = ["OUT_OF_SCOPE_MESSAGE", "out_of_scope_reason"]
