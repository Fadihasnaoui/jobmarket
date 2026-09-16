// Mirrors the Pydantic response models in src/jobmarket/api/models.py and
// jobmarket.cv.workflow.{ExtractedProfileOutput,RecommendationOutput}. Keep in sync by
// hand — there is no schema codegen step in this project yet.

export type DocumentStatus = "parsed" | "parse_error" | "ocr_required";
export type CanonicalExtractionStatus =
  | "success"
  | "insufficient_profile_evidence"
  | "ocr_required"
  | "parse_error";
export type RecommendationMode = "lexical" | "semantic" | "hybrid";

export interface EducationDetail {
  education_status: string | null;
  education_level: string | null;
  education_field: string | null;
  institution: string | null;
  graduation_year: number | null;
  currently_enrolled: boolean | null;
  graduated: boolean | null;
  confidence: number;
  evidence: string[];
  normalized_value: string | null;
}

export interface ExperienceEntry {
  title: string | null;
  employer: string | null;
  start_date: string | null;
  end_date: string | null;
  duration_months: number | null;
  entry_type: string;
  domain: string | null;
  confidence: number;
  evidence: string;
  normalized_value: string | null;
}

export interface ExtractedProfileOutput {
  filename: string;
  mime_type: string | null;
  document_status: DocumentStatus;
  canonical_extraction_status: CanonicalExtractionStatus;
  ocr_required: boolean;
  extraction_confidence_score: number | null;
  extraction_confidence_label: string | null;
  text_quality: number | null;
  section_coverage: number | null;
  education: EducationDetail | null;
  experiences: ExperienceEntry[];
  internships: ExperienceEntry[];
  canonical_skills: string[];
  role_families: string[];
  domains: string[];
  career_level: string;
  extraction_warnings: string[];
  abstention_reason: string | null;
  extraction_method: string;
  extraction_degraded: boolean;
  extraction_degraded_reason: string | null;
}

export interface CvUploadResponse {
  cv_id: string | null;
  expires_at: string;
  document_status: DocumentStatus;
  extraction_status: CanonicalExtractionStatus;
  extraction_method: string;
  extraction_degraded: boolean;
  extraction_degraded_reason: string | null;
  out_of_scope: boolean;
  profile: ExtractedProfileOutput | null;
  warnings: string[];
  message: string | null;
}

export interface RecommendationOutput {
  job_id: number;
  title: string;
  company: string;
  location: string | null;
  contract_type: string | null;
  final_score: number;
  lexical_score: number | null;
  semantic_score: number | null;
  retrieval_source: string;
  skill_score: number;
  role_domain_score: number;
  career_level_compatibility: number;
  matched_skills: string[];
  missing_important_skills: string[];
  explanation: string;
  source_url: string | null;
  posted_at: string | null;
}

export interface MatchesResponse {
  cv_id: string;
  run_id: number;
  mode: RecommendationMode;
  out_of_scope: boolean;
  extraction_degraded: boolean;
  low_confidence: boolean;
  matches: RecommendationOutput[];
  warnings: string[];
  message: string | null;
  elapsed_seconds: number;
}

export interface SkillGapItem {
  canonical_skill: string;
  category: string | null;
  missing_in_top_matches: number;
  overall_job_demand: number;
}

export interface SkillGapResponse {
  cv_id: string;
  run_id: number;
  top_matches_considered: number;
  out_of_scope: boolean;
  gaps: SkillGapItem[];
  warnings: string[];
  message: string | null;
}

export interface SourceStatsOut {
  source: string;
  raw_total: number;
  parsed: number;
  unparsed: number;
}

export interface StatsOverviewResponse {
  raw_jobs: number;
  jobs: number;
  companies: number;
  job_sources: number;
  per_source: SourceStatsOut[];
  reference_run_id: number;
  jobs_in_reference_run: number;
  jobs_with_skills: number;
  skill_coverage_pct: number;
  embedded_jobs: number;
  embedding_coverage_pct: number;
  ontology_skill_count: number;
  distinct_skills_matched: number;
}

export interface SkillDemandItem {
  canonical_skill: string;
  category: string;
  job_count: number;
  pct_of_considered_jobs: number;
}

export interface StatsSkillsResponse {
  run_id: number;
  country: string | null;
  seniority: string | null;
  total_jobs_considered: number;
  skills: SkillDemandItem[];
}

/** Shape of FastAPI's default HTTPException error body: {"detail": ...}. */
export interface ApiErrorBody {
  detail?: string | Record<string, unknown>;
}

export interface ImproveCvRequestBody {
  candidate_name?: string | null;
  contact_info?: string | null;
}

export interface ImprovedExperienceEntry {
  title: string | null;
  employer: string | null;
  start_date: string | null;
  end_date: string | null;
  entry_type: string;
  original_evidence: string;
  improved_description: string;
}

export interface RecommendedSkill {
  canonical_skill: string;
  category: string | null;
  demand_pct_of_matches: number;
  overall_job_demand: number;
}

export interface ImproveCvResponse {
  cv_id: string;
  out_of_scope: boolean;
  extraction_degraded: boolean;
  low_confidence: boolean;
  improved_summary: string;
  improved_experiences: ImprovedExperienceEntry[];
  skills_section: string[];
  recommended_skills_to_develop: RecommendedSkill[];
  changes_explanation: string[];
  source_degraded: boolean;
  source_degraded_reason: string | null;
  docx_base64: string | null;
  warnings: string[];
  message: string | null;
}

export interface QualityReportRequestBody {
  candidate_name?: string | null;
  candidate_title?: string | null;
  contact_info?: string | null;
}

export type QualityCheckStatus = "pass" | "warning" | "fail" | "not_checked";

export interface QualityCheck {
  id: string;
  label: string;
  status: QualityCheckStatus;
  score: number;
  max_score: number;
  message: string;
  suggestion: string | null;
}

export interface SpellingIssue {
  source_label: string;
  original: string;
  corrected: string;
}

export interface UnquantifiedBullet {
  source_label: string;
  text: string;
}

export interface QualityReportResponse {
  cv_id: string;
  out_of_scope: boolean;
  extraction_degraded: boolean;
  low_confidence: boolean;
  overall_score: number;
  checks: QualityCheck[];
  spelling_issues: SpellingIssue[];
  unquantified_bullets: UnquantifiedBullet[];
  recommended_skills_to_develop: RecommendedSkill[];
  dropped_spelling_suggestions: string[];
  source_degraded: boolean;
  source_degraded_reason: string | null;
  docx_base64: string | null;
  warnings: string[];
  message: string | null;
}

export type ChatRoute = "sql" | "rag" | "platform" | "off_topic";
export type ChatRole = "user" | "assistant";

export interface ChatHistoryMessage {
  role: ChatRole;
  content: string;
}

export interface ChatSource {
  label: string;
  url: string | null;
}

export interface ChatJobReference {
  jobId: number;
  title: string;
  company: string;
}

export interface ChatRequest {
  question: string;
  history: ChatHistoryMessage[];
  job_id?: number;
}

export interface ChatResponse {
  route: ChatRoute;
  answer: string;
  sources: ChatSource[];
  data: Record<string, unknown>;
  router_reason: string;
}
