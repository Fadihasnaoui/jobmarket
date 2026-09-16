"""Pydantic request/response models for the jobmarket API — no raw dicts on the wire.

Reuses the already-privacy-safe, already-tested models from `cv/workflow.py`
(`ExtractedProfileOutput`, `RecommendationOutput`) rather than re-deriving equivalent
shapes, so the API surfaces exactly the same honest flags as the CLI.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from jobmarket.cv.improver import ImprovedExperienceEntry, RecommendedSkill
from jobmarket.cv.matching import RecommendationMode
from jobmarket.cv.quality_report import QualityCheck, SpellingIssue, UnquantifiedBullet
from jobmarket.cv.workflow import (
    CanonicalExtractionStatus,
    DocumentStatus,
    ExtractedProfileOutput,
    RecommendationOutput,
)


class CvUploadResponse(BaseModel):
    """Result of parsing, extracting, and embedding one uploaded CV."""

    model_config = ConfigDict(frozen=True)

    cv_id: str | None
    expires_at: datetime
    document_status: DocumentStatus
    extraction_status: CanonicalExtractionStatus
    extraction_method: str
    extraction_degraded: bool
    extraction_degraded_reason: str | None = None
    out_of_scope: bool
    profile: ExtractedProfileOutput | None
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None


class MatchesResponse(BaseModel):
    """Ranked jobs for one previously uploaded CV, with the full explanation payload."""

    model_config = ConfigDict(frozen=True)

    cv_id: str
    run_id: int
    mode: RecommendationMode
    out_of_scope: bool
    extraction_degraded: bool
    low_confidence: bool
    matches: list[RecommendationOutput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None
    elapsed_seconds: float = 0.0


class SkillGapItem(BaseModel):
    """One missing skill aggregated across a CV's top matches, ranked by demand."""

    model_config = ConfigDict(frozen=True)

    canonical_skill: str
    category: str | None = None
    missing_in_top_matches: int
    overall_job_demand: int


class SkillGapResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    cv_id: str
    run_id: int
    top_matches_considered: int
    out_of_scope: bool
    gaps: list[SkillGapItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None


class SourceStatsOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: str
    raw_total: int
    parsed: int
    unparsed: int


class StatsOverviewResponse(BaseModel):
    """Corpus totals, coverage, and per-source breakdown."""

    model_config = ConfigDict(frozen=True)

    raw_jobs: int
    jobs: int
    companies: int
    job_sources: int
    per_source: list[SourceStatsOut] = Field(default_factory=list)
    reference_run_id: int
    jobs_in_reference_run: int
    jobs_with_skills: int
    skill_coverage_pct: float
    embedded_jobs: int
    embedding_coverage_pct: float
    ontology_skill_count: int
    distinct_skills_matched: int


class SkillDemandItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    canonical_skill: str
    category: str
    job_count: int
    pct_of_considered_jobs: float


class StatsSkillsResponse(BaseModel):
    """Skill demand across the reference corpus, optionally filtered."""

    model_config = ConfigDict(frozen=True)

    run_id: int
    country: str | None = None
    seniority: str | None = None
    total_jobs_considered: int
    skills: list[SkillDemandItem] = Field(default_factory=list)


class ImproveCvRequest(BaseModel):
    """Optional, never-stored name/contact for the generated .docx header only.

    Neither field is ever written to `CvStore` — `docx_export.render_improved_cv_docx`
    takes them directly and they exist only for the duration of this one request.
    """

    model_config = ConfigDict(frozen=True)

    candidate_name: str | None = None
    contact_info: str | None = None


class ImproveCvResponse(BaseModel):
    """Result of `/cv/{id}/improve` — reuses `cv/improver.py`'s own Pydantic models
    directly (`ImprovedExperienceEntry`, `RecommendedSkill`) rather than re-deriving
    equivalent shapes, same convention as `ExtractedProfileOutput`/`RecommendationOutput`
    above."""

    model_config = ConfigDict(frozen=True)

    cv_id: str
    out_of_scope: bool
    extraction_degraded: bool
    low_confidence: bool
    improved_summary: str = ""
    improved_experiences: list[ImprovedExperienceEntry] = Field(default_factory=list)
    skills_section: list[str] = Field(default_factory=list)
    recommended_skills_to_develop: list[RecommendedSkill] = Field(default_factory=list)
    changes_explanation: list[str] = Field(default_factory=list)
    source_degraded: bool = False
    source_degraded_reason: str | None = None
    docx_base64: str | None = None
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None


class QualityReportRequest(BaseModel):
    """Optional, never-stored name/title/contact — same contract as `ImproveCvRequest`,
    plus `candidate_title` (which `render_two_column_cv_docx` already accepted but no
    request model exposed yet).

    Unlike `ImproveCvRequest`, `contact_info` isn't only cosmetic here: it also feeds
    `generate_cv_quality_report`'s own "missing contact" check, so a blank value is
    meant to make that check fail honestly, not just leave a doc placeholder.
    """

    model_config = ConfigDict(frozen=True)

    candidate_name: str | None = None
    candidate_title: str | None = None
    contact_info: str | None = None


class QualityReportResponse(BaseModel):
    """Result of `/cv/{id}/quality-report` — reuses `cv/quality_report.py`'s own
    Pydantic models directly (`QualityCheck`, `SpellingIssue`, `UnquantifiedBullet`),
    same convention as `ImproveCvResponse` reusing `cv/improver.py`'s models.

    `docx_base64` here is the two-column template (`render_two_column_cv_docx`), not
    the single-column one `ImproveCvResponse` carries.
    """

    model_config = ConfigDict(frozen=True)

    cv_id: str
    out_of_scope: bool
    extraction_degraded: bool
    low_confidence: bool
    overall_score: int = 0
    checks: list[QualityCheck] = Field(default_factory=list)
    spelling_issues: list[SpellingIssue] = Field(default_factory=list)
    unquantified_bullets: list[UnquantifiedBullet] = Field(default_factory=list)
    recommended_skills_to_develop: list[RecommendedSkill] = Field(default_factory=list)
    dropped_spelling_suggestions: list[str] = Field(default_factory=list)
    source_degraded: bool = False
    source_degraded_reason: str | None = None
    docx_base64: str | None = None
    warnings: list[str] = Field(default_factory=list)
    message: str | None = None


class ChatMessage(BaseModel):
    """One prior turn of conversation history, sent back to the router/RAG answerer
    for context only — never used to answer numeric questions (those are always
    recomputed fresh from the current question's own extracted entities)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str
    history: list[ChatMessage] = Field(default_factory=list)
    job_id: int | None = Field(
        default=None,
        ge=1,
        description="Exact job ID selected from the CV Matches page, when available.",
    )
    run_id: int | None = Field(
        default=None,
        description="Enrichment run for the sql/rag paths; omit for the latest available run "
        "(same convention as /stats/*).",
    )


class ChatSourceOut(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    url: str | None = None


class ChatResponse(BaseModel):
    """Result of `/chat` — `route` and `data` exist specifically for transparency:
    the frontend shows which path answered (sql/rag/platform/off_topic), and `data`
    carries the raw numbers/filters for the sql path so the answer's source is
    inspectable, not just asserted in prose."""

    model_config = ConfigDict(frozen=True)

    route: Literal["sql", "rag", "platform", "off_topic"]
    answer: str
    sources: list[ChatSourceOut] = Field(default_factory=list)
    data: dict[str, object] = Field(default_factory=dict)
    router_reason: str = ""


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatSourceOut",
    "CvUploadResponse",
    "ImproveCvRequest",
    "ImproveCvResponse",
    "MatchesResponse",
    "QualityReportRequest",
    "QualityReportResponse",
    "SkillDemandItem",
    "SkillGapItem",
    "SkillGapResponse",
    "SourceStatsOut",
    "StatsOverviewResponse",
    "StatsSkillsResponse",
]
