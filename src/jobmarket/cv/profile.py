"""Domain models for CV parsing, extraction, and matching."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PARSER_VERSION = "cv-document-parser-v1"
CV_EXTRACTION_VERSION = "cv-deterministic-extractor-v1"
GUARD_VERSION = "cv-llm-guard-v1"
SCORING_VERSION = "cv-job-scorer-v3"
DEFAULT_RECOMMENDATION_RUN_ID = 113
VALIDATION_RUN_WARNING = (
    "Enrichment run 113 contains only 1,000 validation jobs; recommendations are MVP "
    "results over that subset, not the full corpus."
)

CareerLevel = Literal[
    "student",
    "internship",
    "apprenticeship",
    "junior",
    "mid",
    "senior",
    "lead",
    "principal",
    "manager",
    "consultant",
    "unknown",
]
JobType = Literal[
    "cdi",
    "cdd",
    "freelance",
    "stage",
    "apprenticeship",
    "remote",
    "hybrid",
    "on_site",
]


class DocumentTextBlock(BaseModel):
    """Positioned text block extracted from a parsed document page."""

    model_config = ConfigDict(frozen=True)

    page: int
    block_id: str
    x: float
    y: float
    width: float
    height: float
    text: str
    start_offset: int | None = None
    end_offset: int | None = None


class ExtractionTrace(BaseModel):
    """Debug-only provenance for structured CV extraction."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["section", "education", "experience"]
    normalized_value: str
    evidence_text: str
    confidence: float
    page: int | None = None
    source_block_id: str | None = None


class CvSection(BaseModel):
    """Detected semantic section in a parsed CV."""

    model_config = ConfigDict(frozen=True)

    label: str
    normalized_label: str
    start: int
    end: int
    confidence: float = 1.0
    page: int | None = None
    source_block_id: str | None = None


class EducationDetail(BaseModel):
    """Deterministically extracted education evidence."""

    model_config = ConfigDict(frozen=True)

    education_status: str | None = None
    education_level: str | None = None
    education_field: str | None = None
    institution: str | None = None
    graduation_year: int | None = None
    currently_enrolled: bool | None = None
    graduated: bool | None = None
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    page: int | None = None
    source_block_id: str | None = None
    normalized_value: str | None = None


class ExperienceEntry(BaseModel):
    """One deterministic CV experience entry."""

    model_config = ConfigDict(frozen=True)

    title: str | None = None
    employer: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int | None = None
    entry_type: str = "job"
    domain: str | None = None
    confidence: float = 0.0
    evidence: str
    # A short, literal achievement quote for this specific role, when one was found.
    # It is persisted with the already-grounded experience evidence, not the raw CV.
    impact_evidence: str | None = None
    page: int | None = None
    source_block_id: str | None = None
    normalized_value: str | None = None


class RoleInference(BaseModel):
    """Inferred role or domain family with grounded evidence."""

    model_config = ConfigDict(frozen=True)

    value: str
    confidence: float
    evidence: list[str] = Field(default_factory=list)


class ExtractionQuality(BaseModel):
    """Structured confidence and abstention metadata for CV matching."""

    model_config = ConfigDict(frozen=True)

    extraction_confidence_score: float = 1.0
    extraction_confidence_label: Literal["high", "medium", "low"] = "high"
    text_quality: float = 1.0
    section_coverage: float = 0.0
    reliable_skill_count: int = 0
    ambiguous_aliases: list[str] = Field(default_factory=list)
    rejected_ambiguous_aliases: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None


class ExtractionWarning(BaseModel):
    """Non-fatal parser or extraction warning without raw CV content."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class DocumentPage(BaseModel):
    """Text boundaries for one parsed document page when page information is available."""

    model_config = ConfigDict(frozen=True)

    page: int | None
    text: str
    start_offset: int
    end_offset: int


class ParsedDocument(BaseModel):
    """Normalized in-memory representation of parsed CV text."""

    model_config = ConfigDict(frozen=True)

    filename: str
    mime_type: str
    parser_version: str = PARSER_VERSION
    text: str
    page_count: int | None = None
    pages: list[DocumentPage] = Field(default_factory=list)
    blocks: list[DocumentTextBlock] = Field(default_factory=list)
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    file_hash: str
    file_size: int


class EvidenceSpan(BaseModel):
    """Evidence coordinates within the combined CV text and optional page text."""

    model_config = ConfigDict(frozen=True)

    evidence_text: str
    page: int | None = None
    document_start: int
    document_end: int
    page_start: int | None = None
    page_end: int | None = None


class ExtractionVersions(BaseModel):
    """Version stamps for deterministic and guarded extraction."""

    model_config = ConfigDict(frozen=True)

    parser_version: str
    matcher_version: str
    ontology_version: str
    guard_version: str | None = None


ValidationStatus = Literal[
    "deterministic",
    "accepted",
    "rejected_schema",
    "rejected_unknown_skill",
    "rejected_no_evidence",
    "rejected_invalid_offset",
    "rejected_evidence_mismatch",
    "rejected_duplicate",
    "rejected_confidence",
    "rejected_prompt_injection",
    "rejected_unsupported_fact_type",
]


class CvSkill(BaseModel):
    """One skill extracted from a CV with grounded evidence."""

    model_config = ConfigDict(frozen=True)

    canonical_skill: str
    ontology_skill_id: int
    matched_alias: str
    extraction_method: Literal["deterministic", "guarded_llm"]
    evidence: EvidenceSpan
    matcher_version: str
    ontology_version: str
    validation_status: ValidationStatus
    confidence: float | None = None


class ExperienceSummary(BaseModel):
    """Explicitly supported candidate experience summary."""

    model_config = ConfigDict(frozen=True)

    internship_count: int = 0
    professional_experience_count: int = 0
    total_months: int | None = None
    total_years: float | None = None
    evidence: list[str] = Field(default_factory=list)


class CandidateAttributes(BaseModel):
    """Deterministically extracted non-sensitive candidate attributes."""

    model_config = ConfigDict(frozen=True)

    current_role: str | None = None
    career_level: CareerLevel = "unknown"
    education_status: str | None = None
    experience: ExperienceSummary = Field(default_factory=ExperienceSummary)
    preferred_domains: list[str] = Field(default_factory=list)
    programming_languages: list[str] = Field(default_factory=list)
    databases: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    cloud_platforms: list[str] = Field(default_factory=list)
    preferred_country: str | None = None
    preferred_city: str | None = None
    remote_preference: bool | None = None


class CvProfile(BaseModel):
    """Structured CV profile with grounded skills and deterministic attributes."""

    model_config = ConfigDict(frozen=True)

    document: ParsedDocument
    versions: ExtractionVersions
    skills: list[CvSkill]
    candidate: CandidateAttributes = Field(default_factory=CandidateAttributes)
    warnings: list[ExtractionWarning] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    experience_statements: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    sections: list[CvSection] = Field(default_factory=list)
    education_detail: EducationDetail | None = None
    experience_entries: list[ExperienceEntry] = Field(default_factory=list)
    inferred_roles: list[RoleInference] = Field(default_factory=list)
    inferred_domains: list[RoleInference] = Field(default_factory=list)
    extraction_quality: ExtractionQuality = Field(default_factory=ExtractionQuality)
    extraction_traces: list[ExtractionTrace] = Field(default_factory=list)


class GuardedSkillCandidate(BaseModel):
    """Strict provider-neutral LLM candidate shape for one skill claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_type: str
    canonical_skill: str
    ontology_skill_id: int | None = None
    evidence_quote: str
    document_start: int
    document_end: int
    page: int | None = None
    confidence: float


class GuardedCandidateResult(BaseModel):
    """Validation outcome for one LLM candidate."""

    model_config = ConfigDict(frozen=True)

    status: ValidationStatus
    candidate: GuardedSkillCandidate | None = None
    skill: CvSkill | None = None
    reason: str


class GuardedExtractionReport(BaseModel):
    """Grounding report for deterministic plus optional guarded LLM extraction."""

    model_config = ConfigDict(frozen=True)

    parser_version: str
    matcher_version: str
    ontology_version: str
    guard_version: str
    deterministic_skills: list[CvSkill]
    submitted_llm_candidates: int
    accepted_llm_candidates: list[GuardedCandidateResult]
    rejected_candidates: list[GuardedCandidateResult]
    evidence_integrity_violations: list[str]
    document_warnings: list[ExtractionWarning]
    final_skills: list[CvSkill]


class JobAttributeProvenance(BaseModel):
    """Evidence for a deterministic job attribute decision."""

    model_config = ConfigDict(frozen=True)

    value: str
    source: Literal["title", "description", "metadata"]
    evidence: str


class JobAttributes(BaseModel):
    """Deterministically extracted job attributes for compatibility scoring."""

    model_config = ConfigDict(frozen=True)

    career_level: CareerLevel = "unknown"
    experience_min_years: float | None = None
    experience_max_years: float | None = None
    job_types: list[JobType] = Field(default_factory=list)
    remote_mode: JobType | None = None
    country: str | None = None
    city: str | None = None
    domains: list[str] = Field(default_factory=list)
    career_level_provenance: list[JobAttributeProvenance] = Field(default_factory=list)
    job_type_provenance: list[JobAttributeProvenance] = Field(default_factory=list)


class ScoreComponents(BaseModel):
    """Explainable component scores before and after penalties."""

    model_config = ConfigDict(frozen=True)

    skill_score: float
    coverage_component: float = 0.0
    relevance_component: float = 0.0
    richness_component: float = 0.0
    specificity_component: float = 0.0
    skill_profile_confidence: float = 0.0
    specialized_matched_count: int = 0
    generic_matched_count: int = 0
    experience_score: float
    career_level_score: float
    job_type_score: float
    domain_score: float
    location_score: float
    location_status: str = "unknown_candidate_preference"
    weighted_score_before_penalty: float
    penalty_total: float
    lexical_score: float = 0.0
    semantic_score: float | None = None
    alpha: float | None = None
    hybrid_score_before_penalty: float | None = None


class RecommendationPenalty(BaseModel):
    """A deterministic recommendation penalty with an explanation."""

    model_config = ConfigDict(frozen=True)

    code: str
    amount: float
    explanation: str


class JobRecommendation(BaseModel):
    """One deterministic CV-to-job recommendation."""

    model_config = ConfigDict(frozen=True)

    job_id: int
    title: str
    company: str
    location: str | None
    contract_type: str | None = None
    source_url: str | None = None
    posted_at: datetime | None = None
    final_score: float
    coverage_score: float
    cv_overlap_score: float
    cv_skill_count: int
    job_skill_count: int
    matched_skill_count: int
    matched_skills: list[str]
    missing_skills: list[str]
    specialized_matched_skills: list[str] = Field(default_factory=list)
    generic_matched_skills: list[str] = Field(default_factory=list)
    enrichment_run_id: int
    scoring_version: str
    explanation: str
    job_attributes: JobAttributes = Field(default_factory=JobAttributes)
    score_components: ScoreComponents | None = None
    penalties: list[RecommendationPenalty] = Field(default_factory=list)
    experience_gap_years: float | None = None
    tie_break_factors: dict[str, int | float | str] = Field(default_factory=dict)
    semantic_score: float | None = None
    semantic_rank: int | None = None
    retrieval_source: str = "lexical"
    hybrid_score_before_penalties: float | None = None


class CvMatchResult(BaseModel):
    """Complete in-memory CV matching result."""

    model_config = ConfigDict(frozen=True)

    profile: CvProfile
    guard_report: GuardedExtractionReport
    recommendations: list[JobRecommendation]
    enrichment_run_id: int
    validation_run_warning: str | None = None
    status: str = "success"
    message: str | None = None
