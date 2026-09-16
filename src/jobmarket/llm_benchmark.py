# ruff: noqa: E501
"""Controlled offline Groq benchmark for optional job extraction value."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import func, select

from jobmarket.config import get_settings
from jobmarket.cv.attributes import extract_job_attributes
from jobmarket.db.models import Job, JobEnrichmentResult, JobSkill, JobSource, RawJob, Skill
from jobmarket.db.session import async_session_factory
from jobmarket.skills.llm_extractor import _client
from jobmarket.skills.ontology import SkillsOntology, load_ontology, normalize_alias

BENCHMARK_VERSION = "groq-job-benchmark-v1"
DEFAULT_OUTPUT_DIR = Path("reports/llm_benchmark/run_113")
SAMPLE_SEED = 20260724
TRUNCATED_ADZUNA_SHARE = 0.70
NL_IT_DIAGNOSTIC_TARGET = 15
LONG_DESCRIPTION_CHARS = 1500
TECH_TITLE_TERMS = (
    "developer", "engineer", "software", "data", "devops", "cloud", "security",
    "backend", "frontend", "full stack", "machine learning", "ai", "ml", "python",
    "java", "platform", "sre", "architect", "qa", "analyst",
)
MIN_CONFIDENCE = 0.70
MAX_DESCRIPTION_CHARS = 4000
MODEL_PRICING_USD_PER_1M_TOKENS = {"llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79}}
PROMPT_INJECTION_TERMS = ("ignore previous instructions", "system prompt", "developer message")
CareerValue = Literal["internship", "apprenticeship", "junior", "mid", "senior", "lead", "principal", "manager", "unknown"]
WorkModeValue = Literal["remote", "hybrid", "on_site", "unknown"]


class GroqSkillSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    evidence: str
    confidence: float
    explicit: bool = False

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_default(cls, value: object) -> object:
        return 0.0 if value is None else value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be in 0..1")
        return value


class GroqCareerLevel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    value: CareerValue
    evidence: str | None = None
    confidence: float

    @field_validator("value", mode="before")
    @classmethod
    def _canonical_value(cls, value: object) -> str:
        return _canonical_career_value(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_default(cls, value: object) -> object:
        return 0.0 if value is None else value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be in 0..1")
        return value


class GroqExperienceRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")
    minimum_years: float | None = None
    maximum_years: float | None = None
    evidence: str | None = None
    confidence: float

    @field_validator("minimum_years", "maximum_years", mode="before")
    @classmethod
    def _numeric_or_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and normalize_alias(value) in {"", "unknown", "not specified", "n/a"}:
            return None
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_default(cls, value: object) -> object:
        return 0.0 if value is None else value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be in 0..1")
        return value


class GroqWorkMode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    value: WorkModeValue
    evidence: str | None = None
    confidence: float

    @field_validator("value", mode="before")
    @classmethod
    def _canonical_value(cls, value: object) -> str:
        return _canonical_work_mode_value(value)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence_default(cls, value: object) -> object:
        return 0.0 if value is None else value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be in 0..1")
        return value


def _canonical_career_value(value: object) -> str:
    normalized = normalize_alias("" if value is None else str(value))
    if normalized in {"", "unknown", "not specified", "n/a"}:
        return "unknown"
    if normalized in {"internship", "intern", "stage"}:
        return "internship"
    if normalized in {"apprenticeship", "apprentice", "alternance"}:
        return "apprenticeship"
    if normalized in {"junior", "jr", "entry level", "entry-level"}:
        return "junior"
    if normalized in {"mid", "middle", "intermediate"}:
        return "mid"
    if normalized in {"senior", "sr"}:
        return "senior"
    if normalized in {"lead", "tech lead"}:
        return "lead"
    if normalized in {"principal", "staff"}:
        return "principal"
    if normalized in {"manager", "director", "executive", "chief"}:
        return "manager"
    return "unknown"


def _canonical_work_mode_value(value: object) -> str:
    normalized = normalize_alias("" if value is None else str(value))
    if "remote" in normalized or "fully remote" in normalized:
        return "remote"
    if "hybrid" in normalized or "hybride" in normalized or "????????" in normalized:
        return "hybrid"
    if normalized in {"on site", "onsite", "on-site", "office", "in office"}:
        return "on_site"
    return "unknown"


class GroqJobExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skills: list[GroqSkillSuggestion] = Field(default_factory=list)
    career_level: GroqCareerLevel
    experience_requirement: GroqExperienceRequirement
    work_mode: GroqWorkMode


@dataclass(frozen=True)
class BenchmarkJob:
    job_id: int
    title: str
    description: str
    country: str | None = None
    city: str | None = None
    contract_type: str | None = None
    is_remote: bool | None = None
    skill_count: int = 0
    deterministic_skills: tuple[str, ...] = ()
    source: str = ""
    stratum: str = "unassigned"
    diagnostic_group: str = "standard"

    @property
    def source_text(self) -> str:
        return f"{self.title}\n\n{self.description}"


@dataclass(frozen=True)
class GroqCallResult:
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(frozen=True)
class AcceptedSkill:
    job_id: int
    canonical: str
    raw_skill: str
    evidence: str
    confidence: float
    explicit: bool
    category: str


@dataclass(frozen=True)
class RejectedItem:
    job_id: int
    field: str
    value: str
    reason: str
    evidence: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class BenchmarkValidation:
    valid_json: bool
    accepted_skills: list[AcceptedSkill] = field(default_factory=list)
    rejected_items: list[RejectedItem] = field(default_factory=list)
    unmapped_skills: list[RejectedItem] = field(default_factory=list)
    accepted_career_level: CareerValue | None = None
    accepted_experience_min: float | None = None
    accepted_work_mode: WorkModeValue | None = None


@dataclass(frozen=True)
class BenchmarkRecord:
    job: BenchmarkJob
    call: GroqCallResult
    validation: BenchmarkValidation


@dataclass(frozen=True)
class BenchmarkMetrics:
    total_groq_calls: int
    successful_calls: int
    failed_calls: int
    valid_json_rate: float
    guard_acceptance_rate: float
    guard_rejection_rate: float
    mean_accepted_skills_added_per_job: float
    median_accepted_skills_added_per_job: float
    jobs_with_at_least_one_accepted_addition: int
    unresolved_skill_count: int
    career_level_agreement_rate: float
    experience_agreement_rate: float
    work_mode_agreement_rate: float
    average_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_total_cost_usd: float
    estimated_cost_1000_jobs_usd: float
    estimated_cost_full_dataset_usd: float
    stratum_metrics: dict[str, dict[str, float | int]] = field(default_factory=dict)
    accepted_additions_by_existing_skill_status: dict[str, int] = field(default_factory=dict)
    jobs_with_accepted_additions_by_existing_skill_status: dict[str, int] = field(default_factory=dict)
    rejection_reasons_by_stratum: dict[str, dict[str, int]] = field(default_factory=dict)
    seniority_new_values_count: int = 0


@dataclass(frozen=True)
class BenchmarkRunResult:
    records: list[BenchmarkRecord]
    metrics: BenchmarkMetrics
    output_dir: Path


SYSTEM_PROMPT = (
    "Extract evidence-grounded job facts for an offline benchmark. Return only strict JSON. "
    "Never infer unsupported facts. Evidence strings must be exact substrings."
)


def build_user_prompt(job: BenchmarkJob) -> str:
    description = job.description[:MAX_DESCRIPTION_CHARS]
    return (
        f"Job ID: {job.job_id}\nTitle: {job.title}\n\nDescription:\n{description}\n\n"
        "Return JSON with exactly: skills[{name,evidence,confidence,explicit}], "
        "career_level{value,evidence,confidence}, "
        "experience_requirement{minimum_years,maximum_years,evidence,confidence}, "
        "work_mode{value,evidence,confidence}. Unknown is preferred over guessing."
    )


def call_groq_job_extraction(
    job: BenchmarkJob, *, client: Any | None = None, model: str | None = None
) -> GroqCallResult:
    settings = get_settings()
    llm_client = client or _client()
    started = time.perf_counter()
    try:
        response = llm_client.chat.completions.create(
            model=model or settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(job)},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        usage = response.usage
        return GroqCallResult(
            raw_response=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except RateLimitError as exc:
        return _failed_call(started, f"rate_limit: {exc}")
    except (APITimeoutError, TimeoutError) as exc:
        return _failed_call(started, f"timeout: {exc}")
    except APIConnectionError as exc:
        return _failed_call(started, f"connection_error: {exc}")
    except Exception as exc:  # pragma: no cover
        return _failed_call(started, f"llm_error: {exc}")


def _failed_call(started: float, error: str) -> GroqCallResult:
    return GroqCallResult("", latency_ms=(time.perf_counter() - started) * 1000, error=error)


def empty_response_json() -> str:
    return json.dumps(
        {
            "skills": [],
            "career_level": {"value": "unknown", "evidence": None, "confidence": 1.0},
            "experience_requirement": {
                "minimum_years": None,
                "maximum_years": None,
                "evidence": None,
                "confidence": 1.0,
            },
            "work_mode": {"value": "unknown", "evidence": None, "confidence": 1.0},
        },
        sort_keys=True,
    )


def validate_groq_response(
    job: BenchmarkJob, call: GroqCallResult, *, ontology: SkillsOntology | None = None
) -> BenchmarkValidation:
    actual_ontology = ontology or load_ontology()
    if call.error:
        return BenchmarkValidation(False, rejected_items=[RejectedItem(job.job_id, "call", "", call.error)])
    try:
        parsed = GroqJobExtraction.model_validate(json.loads(call.raw_response))
    except (json.JSONDecodeError, ValidationError) as exc:
        return BenchmarkValidation(
            False, rejected_items=[RejectedItem(job.job_id, "json", "", f"invalid_json_schema: {exc}")]
        )
    if _has_prompt_injection(call.raw_response):
        return BenchmarkValidation(
            True, rejected_items=[RejectedItem(job.job_id, "response", "", "prompt_injection")]
        )
    source = job.source_text
    deterministic = set(job.deterministic_skills)
    attrs = extract_job_attributes(
        title=job.title,
        description=job.description,
        contract_type=job.contract_type,
        is_remote=job.is_remote,
        country=job.country,
        city=job.city,
        skill_canonicals=deterministic,
    )
    accepted: list[AcceptedSkill] = []
    rejected: list[RejectedItem] = []
    unmapped: list[RejectedItem] = []
    seen: set[str] = set()
    for skill in parsed.skills:
        canonical = actual_ontology.resolve(skill.name)
        reason = _skill_rejection_reason(skill, source, canonical, seen, deterministic)
        if canonical is None:
            unmapped.append(_reject(job, "skill", skill.name, "unknown_ontology_skill", skill.evidence, skill.confidence))
        elif reason:
            rejected.append(_reject(job, "skill", skill.name, reason, skill.evidence, skill.confidence))
        else:
            seen.add(canonical)
            accepted.append(
                AcceptedSkill(
                    job.job_id, canonical, skill.name, skill.evidence, skill.confidence, skill.explicit,
                    _skill_category(skill),
                )
            )
    career = _accept_career(job, parsed.career_level, attrs, source, rejected)
    exp = _accept_experience(job, parsed.experience_requirement, attrs, source, rejected)
    mode = _accept_work_mode(job, parsed.work_mode, attrs, source, rejected)
    return BenchmarkValidation(True, accepted, rejected, unmapped, career, exp, mode)


def _reject(
    job: BenchmarkJob, field: str, value: str, reason: str, evidence: str | None, confidence: float | None
) -> RejectedItem:
    return RejectedItem(job.job_id, field, value, reason, evidence, confidence)


def _skill_rejection_reason(
    skill: GroqSkillSuggestion, source: str, canonical: str | None, seen: set[str], deterministic: set[str]
) -> str | None:
    if not skill.evidence or skill.evidence not in source:
        return "hallucinated_evidence"
    if skill.confidence < MIN_CONFIDENCE:
        return "low_confidence"
    if canonical is None:
        return "unknown_ontology_skill"
    if canonical in seen:
        return "duplicate_skill"
    if canonical in deterministic:
        return "already_deterministic"
    if _has_prompt_injection(skill.evidence):
        return "prompt_injection"
    return None


def _skill_category(skill: GroqSkillSuggestion) -> str:
    if skill.explicit or normalize_alias(skill.name) in normalize_alias(skill.evidence):
        return "explicit_skill_missed_by_matcher"
    return "implicit_concept_supported_by_evidence"


def _accept_career(
    job: BenchmarkJob, career: GroqCareerLevel, attrs: Any, source: str, rejected: list[RejectedItem]
) -> CareerValue | None:
    if career.value == "unknown":
        return None
    reason = _field_rejection_reason(career.evidence, career.confidence, source)
    if reason is None and attrs.career_level != "unknown" and attrs.career_level != career.value:
        reason = "deterministic_conflict"
    if reason:
        rejected.append(_reject(job, "career_level", career.value, reason, career.evidence, career.confidence))
        return None
    return career.value


def _accept_experience(
    job: BenchmarkJob, exp: GroqExperienceRequirement, attrs: Any, source: str, rejected: list[RejectedItem]
) -> float | None:
    if exp.minimum_years is None and exp.maximum_years is None:
        return None
    value = "" if exp.minimum_years is None else str(exp.minimum_years)
    reason = _field_rejection_reason(exp.evidence, exp.confidence, source)
    if exp.minimum_years is not None and exp.minimum_years < 0:
        reason = "invalid_experience"
    if exp.maximum_years is not None and exp.minimum_years is not None and exp.maximum_years < exp.minimum_years:
        reason = "contradictory_experience"
    if reason is None and attrs.experience_min_years is not None and attrs.experience_min_years != exp.minimum_years:
        reason = "deterministic_conflict"
    if reason:
        rejected.append(_reject(job, "experience", value, reason, exp.evidence, exp.confidence))
        return None
    return exp.minimum_years


def _accept_work_mode(
    job: BenchmarkJob, work: GroqWorkMode, attrs: Any, source: str, rejected: list[RejectedItem]
) -> WorkModeValue | None:
    if work.value == "unknown":
        return None
    reason = _field_rejection_reason(work.evidence, work.confidence, source)
    metadata = "remote" if job.is_remote is True else "on_site" if job.is_remote is False else None
    if reason is None and metadata is not None and metadata != work.value:
        reason = "metadata_conflict"
    if reason is None and attrs.remote_mode is not None and attrs.remote_mode != work.value:
        reason = "deterministic_conflict"
    if reason:
        rejected.append(_reject(job, "work_mode", work.value, reason, work.evidence, work.confidence))
        return None
    return work.value


def _field_rejection_reason(evidence: str | None, confidence: float, source: str) -> str | None:
    if confidence < MIN_CONFIDENCE:
        return "low_confidence"
    if not evidence or evidence not in source:
        return "hallucinated_evidence"
    if _has_prompt_injection(evidence):
        return "prompt_injection"
    return None


def _has_prompt_injection(text: str) -> bool:
    normalized = normalize_alias(text)
    return any(term in normalized for term in PROMPT_INJECTION_TERMS)


def select_sample_from_jobs(jobs: list[BenchmarkJob], sample_size: int) -> list[BenchmarkJob]:
    """Select the Phase-B benchmark sample with deterministic stratification."""
    if sample_size <= 0:
        return []

    truncated_target = round(sample_size * TRUNCATED_ADZUNA_SHARE)
    full_target = sample_size - truncated_target
    nlit_target = min(NL_IT_DIAGNOSTIC_TARGET, truncated_target)

    selected: dict[int, BenchmarkJob] = {}

    truncated_adzuna = [job for job in jobs if job.source == "adzuna"]
    nlit = [job for job in truncated_adzuna if (job.country or "").lower() in {"nl", "it"}]
    non_nlit_adzuna = [job for job in truncated_adzuna if (job.country or "").lower() not in {"nl", "it"}]
    _add_sampled(selected, nlit, nlit_target, "truncated_adzuna", "nl_it_diagnostic", seed_offset=1)
    _add_sampled(
        selected,
        non_nlit_adzuna,
        truncated_target - _count_stratum(selected, "truncated_adzuna"),
        "truncated_adzuna",
        "standard",
        seed_offset=2,
    )
    _add_sampled(
        selected,
        truncated_adzuna,
        truncated_target - _count_stratum(selected, "truncated_adzuna"),
        "truncated_adzuna",
        "standard",
        seed_offset=5,
    )

    full_text = [job for job in jobs if _is_full_text_candidate(job)]
    _add_sampled(selected, full_text, full_target, "full_text", "standard", seed_offset=3)

    if len(selected) < sample_size:
        _add_sampled(selected, jobs, sample_size - len(selected), "fallback", "standard", seed_offset=4)

    return sorted(selected.values(), key=lambda item: item.job_id)[:sample_size]


def _is_full_text_candidate(job: BenchmarkJob) -> bool:
    if len(job.description) >= LONG_DESCRIPTION_CHARS:
        return True
    return job.source == "remoteok" and _is_technical_title(job.title)


def _is_technical_title(title: str) -> bool:
    normalized = normalize_alias(title)
    return any(term in normalized for term in TECH_TITLE_TERMS)


def _add_sampled(
    selected: dict[int, BenchmarkJob],
    candidates: list[BenchmarkJob],
    count: int,
    stratum: str,
    diagnostic_group: str,
    *,
    seed_offset: int,
) -> None:
    if count <= 0:
        return
    ordered = sorted(
        (job for job in candidates if job.job_id not in selected),
        key=lambda job: (_sample_key(job.job_id, seed_offset), job.job_id),
    )
    for job in ordered[:count]:
        group = (
            "nl_it_diagnostic"
            if diagnostic_group == "nl_it_diagnostic" and (job.country or "").lower() in {"nl", "it"}
            else diagnostic_group
        )
        selected[job.job_id] = BenchmarkJob(
            job_id=job.job_id,
            title=job.title,
            description=job.description,
            country=job.country,
            city=job.city,
            contract_type=job.contract_type,
            is_remote=job.is_remote,
            skill_count=job.skill_count,
            deterministic_skills=job.deterministic_skills,
            source=job.source,
            stratum=stratum,
            diagnostic_group=group,
        )


def _sample_key(job_id: int, seed_offset: int) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED + seed_offset}:{job_id}".encode("ascii")).hexdigest()


def _count_stratum(selected: dict[int, BenchmarkJob], stratum: str) -> int:
    return sum(1 for job in selected.values() if job.stratum == stratum)


async def load_run_jobs(run_id: int, *, max_jobs: int | None = None) -> list[BenchmarkJob]:
    async with async_session_factory() as session:
        primary_source = _primary_source_subquery()
        stmt = (
            select(
                Job.id,
                Job.title,
                Job.description,
                Job.country,
                Job.city,
                Job.contract_type,
                Job.is_remote,
                JobEnrichmentResult.skill_count,
                primary_source.c.source,
            )
            .join(JobEnrichmentResult, JobEnrichmentResult.job_id == Job.id)
            .join(primary_source, primary_source.c.job_id == Job.id)
            .where(JobEnrichmentResult.run_id == run_id, JobEnrichmentResult.status == "success")
            .order_by(Job.id)
        )
        if max_jobs is not None:
            stmt = stmt.limit(max_jobs)
        rows = (await session.execute(stmt)).all()
        skill_rows = await session.execute(
            select(JobSkill.job_id, Skill.canonical)
            .join(Skill, Skill.id == JobSkill.skill_id)
            .where(JobSkill.run_id == run_id)
            .order_by(JobSkill.job_id, Skill.canonical)
        )
    if not rows:
        raise ValueError(f"No successful jobs found for run {run_id}")
    skills_by_job: dict[int, list[str]] = {}
    for job_id, canonical in skill_rows.all():
        skills_by_job.setdefault(int(job_id), []).append(str(canonical))
    return [
        BenchmarkJob(
            job_id=int(row.id),
            title=str(row.title),
            description=str(row.description),
            country=row.country,
            city=row.city,
            contract_type=row.contract_type,
            is_remote=row.is_remote,
            skill_count=int(row.skill_count),
            deterministic_skills=tuple(skills_by_job.get(int(row.id), [])),
            source=str(row.source),
        )
        for row in rows
    ]


def _primary_source_subquery() -> Any:
    return (
        select(JobSource.job_id, func.min(RawJob.source).label("source"))
        .join(RawJob, RawJob.id == JobSource.raw_job_id)
        .group_by(JobSource.job_id)
        .subquery("primary_job_source")
    )


async def run_groq_benchmark(
    *, run_id: int = 113, sample_size: int = 100, output_dir: Path = DEFAULT_OUTPUT_DIR,
    model: str | None = None, max_jobs: int | None = None, dry_run: bool = False,
    resume: bool = False, offline_report: bool = False, client: Any | None = None,
) -> BenchmarkRunResult:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    settings = get_settings()
    if not dry_run and not offline_report and not settings.llm_api_key and client is None:
        raise RuntimeError("LLM_API_KEY is not set; use --dry-run or --offline-report")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = select_sample_from_jobs(await load_run_jobs(run_id, max_jobs=max_jobs), sample_size)
    _write_sample(output_dir / "sample_job_ids.csv", jobs)
    ontology = load_ontology()
    records: list[BenchmarkRecord] = []
    for job in jobs:
        cache_path = cache_dir / f"job_{job.job_id}.json"
        if offline_report or (resume and cache_path.exists()):
            call = _read_cached_call(cache_path)
        elif dry_run:
            call = GroqCallResult(empty_response_json(), latency_ms=0.0)
            _write_cached_call(cache_path, call)
        else:
            call = call_groq_job_extraction(job, client=client, model=model)
            _write_cached_call(cache_path, call)
        records.append(BenchmarkRecord(job, call, validate_groq_response(job, call, ontology=ontology)))
    metrics = compute_metrics(records)
    write_reports(output_dir, records, metrics, run_id=run_id, model=model or settings.llm_model)
    return BenchmarkRunResult(records, metrics, output_dir)


def _read_cached_call(path: Path) -> GroqCallResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    return GroqCallResult(
        str(data.get("raw_response", "")), int(data.get("prompt_tokens", 0)),
        int(data.get("completion_tokens", 0)), float(data.get("latency_ms", 0.0)), data.get("error"),
    )


def _write_cached_call(path: Path, call: GroqCallResult) -> None:
    path.write_text(json.dumps(call.__dict__, indent=2, sort_keys=True), encoding="utf-8")


def _write_sample(path: Path, jobs: list[BenchmarkJob]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "job_id",
                "source",
                "country",
                "stratum",
                "diagnostic_group",
                "skill_count",
                "description_length",
                "title",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_id": job.job_id,
                    "source": job.source,
                    "country": job.country or "",
                    "stratum": job.stratum,
                    "diagnostic_group": job.diagnostic_group,
                    "skill_count": job.skill_count,
                    "description_length": len(job.description),
                    "title": job.title,
                }
            )


def compute_metrics(records: list[BenchmarkRecord]) -> BenchmarkMetrics:
    total = len(records)
    successful = sum(1 for r in records if r.call.error is None)
    valid = sum(1 for r in records if r.validation.valid_json)
    accepted_counts = [len(r.validation.accepted_skills) for r in records]
    rejected_count = sum(len(r.validation.rejected_items) for r in records)
    accepted_count = sum(accepted_counts)
    unresolved = sum(len(r.validation.unmapped_skills) for r in records)
    latencies = [r.call.latency_ms for r in records]
    prompt_tokens = sum(r.call.prompt_tokens for r in records)
    completion_tokens = sum(r.call.completion_tokens for r in records)
    cost = estimate_cost(prompt_tokens, completion_tokens)
    attempts = accepted_count + rejected_count + unresolved
    return BenchmarkMetrics(
        sum(1 for r in records if r.call.prompt_tokens or r.call.completion_tokens or r.call.latency_ms > 0), successful, total - successful, _rate(valid, total), _rate(accepted_count, attempts),
        _rate(rejected_count + unresolved, attempts), statistics.fmean(accepted_counts) if accepted_counts else 0.0,
        statistics.median(accepted_counts) if accepted_counts else 0.0, sum(1 for c in accepted_counts if c),
        unresolved, _agreement(records, "career"), _agreement(records, "experience"), _agreement(records, "work_mode"),
        statistics.fmean(latencies) if latencies else 0.0, _percentile(latencies, 50), _percentile(latencies, 95),
        prompt_tokens, completion_tokens, cost, (cost / total * 1000) if total else 0.0,
        (cost / total * 35968) if total else 0.0,
        _stratum_metrics(records), _accepted_by_existing_skill_status(records),
        _jobs_with_accepted_by_existing_skill_status(records), _rejection_reasons_by_stratum(records),
        _seniority_new_values_count(records),
    )


def _stratum_metrics(records: list[BenchmarkRecord]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for stratum in sorted({record.job.stratum for record in records}):
        subset = [record for record in records if record.job.stratum == stratum]
        matcher_counts = [record.job.skill_count for record in subset]
        llm_counts = [record.job.skill_count + len(record.validation.accepted_skills) for record in subset]
        accepted = sum(len(record.validation.accepted_skills) for record in subset)
        rejected = sum(len(record.validation.rejected_items) + len(record.validation.unmapped_skills) for record in subset)
        prompt = sum(record.call.prompt_tokens for record in subset)
        completion = sum(record.call.completion_tokens for record in subset)
        output[stratum] = {
            "jobs": len(subset),
            "matcher_mean_skills_per_job": statistics.fmean(matcher_counts) if matcher_counts else 0.0,
            "matcher_plus_llm_mean_skills_per_job": statistics.fmean(llm_counts) if llm_counts else 0.0,
            "accepted_additions": accepted,
            "rejected_or_unmapped": rejected,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "estimated_cost_usd": estimate_cost(prompt, completion),
            "p50_latency_ms": _percentile([record.call.latency_ms for record in subset], 50),
            "p95_latency_ms": _percentile([record.call.latency_ms for record in subset], 95),
        }
    return output


def _accepted_by_existing_skill_status(records: list[BenchmarkRecord]) -> dict[str, int]:
    return {
        "zero_skill_jobs": sum(len(record.validation.accepted_skills) for record in records if record.job.skill_count == 0),
        "nonzero_skill_jobs": sum(len(record.validation.accepted_skills) for record in records if record.job.skill_count > 0),
    }


def _jobs_with_accepted_by_existing_skill_status(records: list[BenchmarkRecord]) -> dict[str, int]:
    return {
        "zero_skill_jobs": sum(1 for record in records if record.job.skill_count == 0 and record.validation.accepted_skills),
        "nonzero_skill_jobs": sum(1 for record in records if record.job.skill_count > 0 and record.validation.accepted_skills),
    }


def _rejection_reasons_by_stratum(records: list[BenchmarkRecord]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = output.setdefault(record.job.stratum, {})
        for item in [*record.validation.rejected_items, *record.validation.unmapped_skills]:
            bucket[item.reason] = bucket.get(item.reason, 0) + 1
    return {stratum: dict(sorted(reasons.items())) for stratum, reasons in sorted(output.items())}


def _seniority_new_values_count(records: list[BenchmarkRecord]) -> int:
    count = 0
    for record in records:
        attrs = extract_job_attributes(
            title=record.job.title, description=record.job.description, contract_type=record.job.contract_type,
            is_remote=record.job.is_remote, country=record.job.country, city=record.job.city,
            skill_canonicals=set(record.job.deterministic_skills),
        )
        if attrs.career_level == "unknown" and record.validation.accepted_career_level is not None:
            count += 1
    return count


def _agreement(records: list[BenchmarkRecord], field: str) -> float:
    if not records:
        return 0.0
    count = 0
    for record in records:
        attrs = extract_job_attributes(
            title=record.job.title, description=record.job.description, contract_type=record.job.contract_type,
            is_remote=record.job.is_remote, country=record.job.country, city=record.job.city,
            skill_canonicals=set(record.job.deterministic_skills),
        )
        if field == "career":
            candidate = record.validation.accepted_career_level
            count += int(candidate is None or candidate == attrs.career_level)
        elif field == "experience":
            candidate_exp = record.validation.accepted_experience_min
            count += int(candidate_exp is None or candidate_exp == attrs.experience_min_years)
        elif field == "work_mode":
            candidate_mode = record.validation.accepted_work_mode
            count += int(candidate_mode is None or candidate_mode == attrs.remote_mode)
    return count / len(records)


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str | None = None) -> float:
    actual_model = model or get_settings().llm_model
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(actual_model, {"input": 0.0, "output": 0.0})
    return round(prompt_tokens / 1_000_000 * pricing["input"] + completion_tokens / 1_000_000 * pricing["output"], 6)


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return ordered[index]


def write_reports(output_dir: Path, records: list[BenchmarkRecord], metrics: BenchmarkMetrics, *, run_id: int, model: str) -> None:
    payload = {"benchmark_version": BENCHMARK_VERSION, "run_id": run_id, "model": model, "sample": _sample_summary(records), "metrics": metrics.__dict__}
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "accepted_implicit_skills.csv", _accepted_rows(records))
    _write_csv(output_dir / "rejected_suggestions.csv", _rejected_rows(records))
    _write_csv(output_dir / "unmapped_skills.csv", _unmapped_rows(records))
    _write_csv(output_dir / "manual_review_template.csv", _manual_rows(records[:20]))
    _write_csv(output_dir / "seniority_spot_check.csv", _seniority_spot_check_rows(records))
    (output_dir / "decision_report.md").write_text(_decision_report(metrics), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        rows = [{}]
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _accepted_rows(records: list[BenchmarkRecord]) -> list[dict[str, object]]:
    return [
        skill.__dict__
        | {
            "source": r.job.source,
            "country": r.job.country or "",
            "stratum": r.job.stratum,
            "diagnostic_group": r.job.diagnostic_group,
            "existing_skill_status": "zero_skill_jobs" if r.job.skill_count == 0 else "nonzero_skill_jobs",
            "human_correct": "",
            "reviewer_notes": "",
        }
        for r in records
        for skill in r.validation.accepted_skills
    ]


def _rejected_rows(records: list[BenchmarkRecord]) -> list[dict[str, object]]:
    return [
        item.__dict__
        | {
            "source": r.job.source,
            "country": r.job.country or "",
            "stratum": r.job.stratum,
            "diagnostic_group": r.job.diagnostic_group,
        }
        for r in records
        for item in r.validation.rejected_items
    ]


def _unmapped_rows(records: list[BenchmarkRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for record in records:
        for item in record.validation.unmapped_skills:
            counts[item.value] = counts.get(item.value, 0) + 1
            rows.append(
                item.__dict__
                | {
                    "source": record.job.source,
                    "country": record.job.country or "",
                    "stratum": record.job.stratum,
                    "diagnostic_group": record.job.diagnostic_group,
                    "occurrence_count": counts[item.value],
                }
            )
    return sorted(rows, key=_unmapped_sort_key)


def _unmapped_sort_key(row: dict[str, object]) -> tuple[str, int]:
    raw_job_id = row.get("job_id", 0)
    job_id = raw_job_id if isinstance(raw_job_id, int) else 0
    return str(row.get("value", "")), job_id


def _manual_rows(records: list[BenchmarkRecord]) -> list[dict[str, object]]:
    return [
        {
            "job_id": r.job.job_id,
            "title": r.job.title,
            "source": r.job.source,
            "country": r.job.country or "",
            "stratum": r.job.stratum,
            "skill_count": r.job.skill_count,
            "accepted_skill_correct": "",
            "career_level_correct": "",
            "experience_correct": "",
            "work_mode_correct": "",
            "reviewer_notes": "",
        }
        for r in records
    ]


def _seniority_spot_check_rows(records: list[BenchmarkRecord]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        attrs = extract_job_attributes(
            title=record.job.title,
            description=record.job.description,
            contract_type=record.job.contract_type,
            is_remote=record.job.is_remote,
            country=record.job.country,
            city=record.job.city,
            skill_canonicals=set(record.job.deterministic_skills),
        )
        if attrs.career_level != "unknown" or record.validation.accepted_career_level is None:
            continue
        parsed = _parsed_extraction(record)
        rows.append(
            {
                "job_id": record.job.job_id,
                "title": record.job.title,
                "source": record.job.source,
                "country": record.job.country or "",
                "stratum": record.job.stratum,
                "diagnostic_group": record.job.diagnostic_group,
                "llm_career_level": record.validation.accepted_career_level,
                "llm_evidence": parsed.career_level.evidence if parsed else "",
                "llm_confidence": parsed.career_level.confidence if parsed else "",
                "human_correct": "",
                "reviewer_notes": "",
            }
        )
    return rows[:15]


def _parsed_extraction(record: BenchmarkRecord) -> GroqJobExtraction | None:
    try:
        return GroqJobExtraction.model_validate(json.loads(record.call.raw_response))
    except (json.JSONDecodeError, ValidationError):
        return None


def _sample_summary(records: list[BenchmarkRecord]) -> dict[str, object]:
    by_stratum: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_country: dict[str, int] = {}
    by_skill_status: dict[str, int] = {"zero_skill_jobs": 0, "nonzero_skill_jobs": 0}
    diagnostic_nl_it = 0
    for record in records:
        by_stratum[record.job.stratum] = by_stratum.get(record.job.stratum, 0) + 1
        by_source[record.job.source] = by_source.get(record.job.source, 0) + 1
        country = record.job.country or "(unknown)"
        by_country[country] = by_country.get(country, 0) + 1
        status = "zero_skill_jobs" if record.job.skill_count == 0 else "nonzero_skill_jobs"
        by_skill_status[status] += 1
        if record.job.diagnostic_group == "nl_it_diagnostic":
            diagnostic_nl_it += 1
    return {
        "seed": SAMPLE_SEED,
        "by_stratum": dict(sorted(by_stratum.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_country": dict(sorted(by_country.items())),
        "by_existing_skill_status": by_skill_status,
        "nl_it_diagnostic_jobs": diagnostic_nl_it,
    }


def _decision_report(metrics: BenchmarkMetrics) -> str:
    return (
        "# Groq Benchmark Decision Report\n\n"
        "Do not claim precision or recall without human labels.\n\n"
        f"- total Groq calls: {metrics.total_groq_calls}\n"
        f"- valid JSON rate: {metrics.valid_json_rate:.1%}\n"
        f"- guard acceptance rate: {metrics.guard_acceptance_rate:.1%}\n"
        f"- guard rejection rate: {metrics.guard_rejection_rate:.1%}\n"
        f"- mean accepted skills added per job: {metrics.mean_accepted_skills_added_per_job:.2f}\n"
        f"- unresolved skill count: {metrics.unresolved_skill_count}\n"
        f"- accepted additions from zero-skill jobs: {metrics.accepted_additions_by_existing_skill_status.get('zero_skill_jobs', 0)}\n"
        f"- accepted additions from nonzero-skill jobs: {metrics.accepted_additions_by_existing_skill_status.get('nonzero_skill_jobs', 0)}\n"
        f"- seniority new values where deterministic was unknown: {metrics.seniority_new_values_count}\n"
        f"- estimated total cost USD: {metrics.estimated_total_cost_usd:.6f}\n"
        f"- estimated cost for 1,000 jobs USD: {metrics.estimated_cost_1000_jobs_usd:.6f}\n"
        f"- estimated cost for full dataset USD: {metrics.estimated_cost_full_dataset_usd:.6f}\n"
    )


__all__ = [
    "BenchmarkJob", "GroqCallResult", "GroqJobExtraction", "call_groq_job_extraction",
    "compute_metrics", "estimate_cost", "run_groq_benchmark", "select_sample_from_jobs",
    "validate_groq_response",
]
