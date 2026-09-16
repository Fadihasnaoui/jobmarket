import type {
  ApiErrorBody,
  ChatRequest,
  ChatResponse,
  CvUploadResponse,
  ImproveCvRequestBody,
  ImproveCvResponse,
  MatchesResponse,
  QualityReportRequestBody,
  QualityReportResponse,
  RecommendationMode,
  SkillGapResponse,
  StatsOverviewResponse,
  StatsSkillsResponse,
} from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8123";

/** Thrown for any non-2xx response, carrying the server's own error detail when present. */
export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (typeof body.detail === "string") return body.detail;
    if (body.detail) return JSON.stringify(body.detail);
  } catch {
    // Response wasn't JSON — fall through to the generic message below.
  }
  return `Request failed with status ${response.status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(
      0,
      "Could not reach the jobmarket API. Is it running (uvicorn jobmarket.api.main:app)?",
    );
  }
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }
  return (await response.json()) as T;
}

export async function uploadCv(file: File): Promise<CvUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<CvUploadResponse>("/cv/upload", { method: "POST", body: formData });
}

export interface MatchesParams {
  mode?: RecommendationMode;
  limit?: number;
  alpha?: number;
  runId?: number;
  country?: string;
  contractType?: string;
  remote?: boolean;
}

export async function getMatches(cvId: string, params: MatchesParams = {}): Promise<MatchesResponse> {
  const query = new URLSearchParams();
  if (params.mode) query.set("mode", params.mode);
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  if (params.alpha !== undefined) query.set("alpha", String(params.alpha));
  if (params.runId !== undefined) query.set("run_id", String(params.runId));
  if (params.country) query.set("country", params.country);
  if (params.contractType) query.set("contract_type", params.contractType);
  if (params.remote !== undefined) query.set("remote", String(params.remote));
  return request<MatchesResponse>(`/cv/${cvId}/matches?${query.toString()}`);
}

export interface SkillGapParams {
  top?: number;
  runId?: number;
  mode?: RecommendationMode;
}

export async function getSkillGap(cvId: string, params: SkillGapParams = {}): Promise<SkillGapResponse> {
  const query = new URLSearchParams();
  if (params.top !== undefined) query.set("top", String(params.top));
  if (params.runId !== undefined) query.set("run_id", String(params.runId));
  if (params.mode) query.set("mode", params.mode);
  return request<SkillGapResponse>(`/cv/${cvId}/skill-gap?${query.toString()}`);
}

export async function getStatsOverview(runId?: number): Promise<StatsOverviewResponse> {
  const query = new URLSearchParams();
  if (runId !== undefined) query.set("run_id", String(runId));
  return request<StatsOverviewResponse>(`/stats/overview?${query.toString()}`);
}

export interface StatsSkillsParams {
  runId?: number;
  country?: string;
  seniority?: string;
  limit?: number;
}

export async function getStatsSkills(params: StatsSkillsParams = {}): Promise<StatsSkillsResponse> {
  const query = new URLSearchParams();
  if (params.runId !== undefined) query.set("run_id", String(params.runId));
  if (params.country) query.set("country", params.country);
  if (params.seniority) query.set("seniority", params.seniority);
  if (params.limit !== undefined) query.set("limit", String(params.limit));
  return request<StatsSkillsResponse>(`/stats/skills?${query.toString()}`);
}

export interface ImproveCvParams {
  runId?: number;
  mode?: RecommendationMode;
}

/** candidateName/contactInfo are sent for this one request only — never persisted
 * anywhere client- or server-side beyond generating this response's .docx bytes. */
export async function improveCv(
  cvId: string,
  body: ImproveCvRequestBody,
  params: ImproveCvParams = {},
): Promise<ImproveCvResponse> {
  const query = new URLSearchParams();
  if (params.runId !== undefined) query.set("run_id", String(params.runId));
  if (params.mode) query.set("mode", params.mode);
  return request<ImproveCvResponse>(`/cv/${cvId}/improve?${query.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export interface QualityReportParams {
  runId?: number;
  mode?: RecommendationMode;
}

/** candidateName/contactInfo are sent for this one request only — same never-stored
 * contract as `improveCv`; here they also drive the "missing contact" check itself. */
export async function getQualityReport(
  cvId: string,
  body: QualityReportRequestBody,
  params: QualityReportParams = {},
): Promise<QualityReportResponse> {
  const query = new URLSearchParams();
  if (params.runId !== undefined) query.set("run_id", String(params.runId));
  if (params.mode) query.set("mode", params.mode);
  return request<QualityReportResponse>(`/cv/${cvId}/quality-report?${query.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function sendChatMessage(body: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
