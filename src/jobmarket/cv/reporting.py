"""Formatting helpers for CV matching results."""

from __future__ import annotations

import json

from jobmarket.cv.profile import CvMatchResult


def cv_match_result_to_json(result: CvMatchResult) -> str:
    """Render a CV match result as deterministic JSON without raw CV text."""
    payload = _privacy_safe_payload(result)
    return json.dumps(payload, indent=2, sort_keys=True)


def format_cv_match_result(result: CvMatchResult) -> str:
    """Render a human-readable privacy-safe CV matching result."""
    lines = [
        "CV match result",
        "===============",
        f"filename:              {result.profile.document.filename}",
        f"mime type:             {result.profile.document.mime_type}",
        f"parser version:        {result.profile.versions.parser_version}",
        f"matcher version:       {result.profile.versions.matcher_version}",
        f"ontology version:      {result.profile.versions.ontology_version}",
        f"guard version:         {result.guard_report.guard_version}",
        f"enrichment run:        {result.enrichment_run_id}",
    ]
    lines.append(f"status:                {result.status}")
    if result.message:
        lines.append(f"message:               {result.message}")
    quality = result.profile.extraction_quality
    detected_sections = ", ".join(s.normalized_label for s in result.profile.sections) or "none"
    rejected_aliases = ", ".join(quality.rejected_ambiguous_aliases) or "none"
    lines.extend(
        [
            f"extraction confidence: {quality.extraction_confidence_label} "
            f"({quality.extraction_confidence_score:.2f})",
            f"text quality:          {quality.text_quality:.2f}",
            f"section coverage:      {quality.section_coverage:.2f}",
            f"detected sections:     {detected_sections}",
            f"rejected aliases:      {rejected_aliases}",
        ]
    )
    if quality.abstention_reason:
        lines.append(f"abstention reason:     {quality.abstention_reason}")
    if result.validation_run_warning:
        lines.extend(["", f"WARNING: {result.validation_run_warning}"])
    if result.profile.document.warnings:
        lines.extend(["", "Parser warnings", "---------------"])
        for warning in result.profile.document.warnings:
            lines.append(f"{warning.code}: {warning.message}")
    candidate = result.profile.candidate
    experience_years = candidate.experience.total_years
    experience_label = experience_years if experience_years is not None else "unknown"
    lines.extend(
        [
            "",
            "Candidate attributes",
            "--------------------",
            f"role:                  {candidate.current_role or 'unknown'}",
            f"career level:          {candidate.career_level}",
            f"education status:      {candidate.education_status or 'unknown'}",
            f"experience years:      {experience_label}",
            f"internships:           {candidate.experience.internship_count}",
            f"professional entries:  {candidate.experience.professional_experience_count}",
            f"domains:               {', '.join(candidate.preferred_domains) or 'unknown'}",
        ]
    )
    if result.profile.education_detail is not None:
        edu = result.profile.education_detail
        lines.extend(
            [
                "",
                "Education evidence",
                "------------------",
                f"status:                {edu.education_status or 'unknown'}",
                f"level:                 {edu.education_level or 'unknown'}",
                f"field:                 {edu.education_field or 'unknown'}",
                f"confidence:            {edu.confidence:.2f}",
                f"evidence:              {', '.join(edu.evidence) or 'none'}",
            ]
        )
    if result.profile.experience_entries:
        lines.extend(["", "Experience evidence", "-------------------"])
        for entry in result.profile.experience_entries:
            lines.append(
                f"- {entry.entry_type}: {entry.title or 'unknown'} "
                f"{entry.start_date or '?'}-{entry.end_date or '?'} "
                f"months={entry.duration_months or 'unknown'} domain={entry.domain or 'unknown'}"
            )
    if result.profile.inferred_roles:
        lines.extend(["", "Role evidence", "-------------"])
        for role in result.profile.inferred_roles[:5]:
            lines.append(
                f"- {role.value} confidence={role.confidence:.2f}: {', '.join(role.evidence)}"
            )
    lines.extend(["", "Extracted CV skills", "-------------------"])
    if not result.profile.skills:
        lines.append("(none)")
    for skill in result.profile.skills:
        page = "n/a" if skill.evidence.page is None else str(skill.evidence.page)
        lines.append(
            f"- {skill.canonical_skill} via {skill.matched_alias!r} "
            f"at {skill.evidence.document_start}-{skill.evidence.document_end} page={page}"
        )
    if result.guard_report.rejected_candidates:
        lines.extend(["", "Rejected guarded candidates", "---------------------------"])
        for item in result.guard_report.rejected_candidates:
            lines.append(f"- {item.status}: {item.reason}")
    lines.extend(["", "Ranked jobs", "-----------"])
    if not result.recommendations:
        lines.append("(none)")
    for index, job in enumerate(result.recommendations, start=1):
        matched = ", ".join(job.matched_skills) or "none"
        missing = ", ".join(job.missing_skills[:8]) or "none"
        if len(job.missing_skills) > 8:
            missing += ", ..."
        components = job.score_components
        skill_score = components.skill_score if components else 0
        experience_score = components.experience_score if components else 0
        career_score = components.career_level_score if components else 0
        domain_score = components.domain_score if components else 0
        job_type_score = components.job_type_score if components else 0
        location_score = components.location_score if components else 0
        specialized = ", ".join(job.specialized_matched_skills) or "none"
        generic = ", ".join(job.generic_matched_skills) or "none"
        coverage = components.coverage_component if components else job.coverage_score
        richness = components.skill_profile_confidence if components else 0
        location_status = components.location_status if components else "unknown"
        lines.extend(
            [
                f"{index}. job {job.job_id}: {job.title} ({job.company})",
                f"   score={job.final_score:.1f} skill={skill_score:.2f} "
                f"coverage={coverage:.2f} matched={job.matched_skill_count}/{job.job_skill_count}",
                f"   semantic={_format_optional_score(job.semantic_score)} "
                f"hybrid_before_penalty="
                f"{_format_optional_score(job.hybrid_score_before_penalties)} "
                f"retrieval_source={job.retrieval_source}",
                f"   specialized_matches={len(job.specialized_matched_skills)} "
                f"generic_matches={len(job.generic_matched_skills)} "
                f"skill_profile_confidence={richness:.2f}",
                f"   experience={experience_score:.2f} career={career_score:.2f} "
                f"job_type={job_type_score:.2f}",
                f"   domain={domain_score:.2f} location={location_score:.2f} "
                f"location_status={location_status}",
                f"   matched: {matched}",
                f"   specialized: {specialized}",
                f"   generic: {generic}",
                f"   missing: {missing}",
                f"   penalties: {_format_penalties(job)}",
                f"   tie_break: {_format_tie_break(job)}",
                f"   provenance: {_format_job_provenance(job)}",
                f"   {job.explanation}",
            ]
        )
    return "\n".join(lines)


def _format_optional_score(value: object) -> object:
    return value if value is not None else "n/a"


def _format_penalties(job: object) -> str:
    penalties = getattr(job, "penalties", [])
    if not penalties:
        return "none"
    return "; ".join(f"{penalty.code}({penalty.amount:.2f})" for penalty in penalties)


def _format_tie_break(job: object) -> str:
    factors = getattr(job, "tie_break_factors", {}) or {}
    if not factors:
        return "none"
    keys = [
        "skill_coverage",
        "specialized_skill_overlap_count",
        "total_matched_skill_count",
        "job_skill_richness",
        "career_compatibility",
        "experience_compatibility",
        "job_id",
    ]
    return ", ".join(f"{key}={factors.get(key)}" for key in keys)


def _format_job_provenance(job: object) -> str:
    attrs = getattr(job, "job_attributes", None)
    if attrs is None:
        return "none"
    career = _format_provenance_items(getattr(attrs, "career_level_provenance", []))
    job_type = _format_provenance_items(getattr(attrs, "job_type_provenance", []))
    return f"career=[{career}], job_type=[{job_type}]"


def _format_provenance_items(items: object) -> str:
    if not isinstance(items, list) or not items:
        return "none"
    return "; ".join(f"{item.value} from {item.source}: {item.evidence!r}" for item in items)


def _privacy_safe_payload(result: CvMatchResult) -> dict[str, object]:
    data = result.model_dump(mode="json")
    profile = data["profile"]
    if isinstance(profile, dict):
        document = profile.get("document")
        if isinstance(document, dict):
            document.pop("text", None)
            for page in document.get("pages", []) or []:
                if isinstance(page, dict):
                    page.pop("text", None)
    guard = data.get("guard_report")
    if isinstance(guard, dict):
        for key in (
            "deterministic_skills",
            "accepted_llm_candidates",
            "rejected_candidates",
            "final_skills",
        ):
            _strip_nested_raw_text(guard.get(key))
    return data


def _strip_nested_raw_text(value: object) -> None:
    if isinstance(value, dict):
        value.pop("text", None)
        for child in value.values():
            _strip_nested_raw_text(child)
    elif isinstance(value, list):
        for child in value:
            _strip_nested_raw_text(child)


__all__ = ["cv_match_result_to_json", "format_cv_match_result"]
