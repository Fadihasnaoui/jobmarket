"""SQLAlchemy ORM models for raw and clean data layers."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates
from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType[list[float]]):
    """Minimal pgvector SQLAlchemy type for DDL and metadata."""

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **_kw: object) -> str:
        return f"vector({self.dimension})"


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class RawJob(Base):
    """Append-only store of verbatim API payloads."""

    __tablename__ = "raw_jobs"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_raw_jobs_source_source_id"),
        Index("ix_raw_jobs_parsed_at", "parsed_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    job_sources: Mapped[list["JobSource"]] = relationship(back_populates="raw_job")


class IngestRun(Base):
    """Audit log for each ingest execution."""

    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    rows_fetched: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    rows_inserted: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class Company(Base):
    """Normalized company entities."""

    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    name_norm: Mapped[str] = mapped_column(Text, nullable=False, unique=True)

    jobs: Mapped[list["Job"]] = relationship(back_populates="company")


class Job(Base):
    """Deduplicated, normalized job postings."""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_company_id", "company_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    company_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("companies.id"), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    title_norm: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    location_raw: Mapped[str | None] = mapped_column(Text, nullable=True)
    country: Mapped[str | None] = mapped_column(Text, nullable=True)
    city: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_remote: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    contract_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    salary_min: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    salary_max: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    salary_currency: Mapped[str | None] = mapped_column(Text, nullable=True)
    salary_period: Mapped[str | None] = mapped_column(Text, nullable=True)
    salary_is_predicted: Mapped[bool] = mapped_column(
        Boolean, server_default="false", nullable=False
    )
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    company: Mapped["Company"] = relationship(back_populates="jobs")
    job_sources: Mapped[list["JobSource"]] = relationship(back_populates="job")
    job_skills: Mapped[list["JobSkill"]] = relationship(back_populates="job")
    enrichment_failures: Mapped[list["EnrichmentFailure"]] = relationship(back_populates="job")
    enrichment_results: Mapped[list["JobEnrichmentResult"]] = relationship(back_populates="job")


class JobSource(Base):
    """Many-to-many link between deduplicated jobs and raw source records."""

    __tablename__ = "job_sources"

    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id"), primary_key=True)
    raw_job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("raw_jobs.id"), primary_key=True)

    job: Mapped["Job"] = relationship(back_populates="job_sources")
    raw_job: Mapped["RawJob"] = relationship(back_populates="job_sources")


class Skill(Base):
    """Canonical skill from the hand-curated skills ontology."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    canonical: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("skills.id"), nullable=True
    )

    job_skills: Mapped[list["JobSkill"]] = relationship(back_populates="skill")


class EnrichmentRun(Base):
    """Observable execution record for matcher or LLM enrichment work.

    A resumed execution continues the same run_id, preserving run history while
    keeping inserts idempotent within that run.
    """

    __tablename__ = "enrichment_runs"
    __table_args__ = (
        CheckConstraint(
            "run_type IN ('matcher', 'llm_experiment')",
            name="ck_enrichment_runs_run_type",
        ),
        CheckConstraint(
            "status IN ('running', 'success', 'partial', 'failed')",
            name="ck_enrichment_runs_status",
        ),
        CheckConstraint("jobs_total >= 0", name="ck_enrichment_runs_jobs_total_nonnegative"),
        CheckConstraint(
            "jobs_processed >= 0",
            name="ck_enrichment_runs_jobs_processed_nonnegative",
        ),
        CheckConstraint(
            "jobs_succeeded >= 0",
            name="ck_enrichment_runs_jobs_succeeded_nonnegative",
        ),
        CheckConstraint(
            "jobs_failed >= 0",
            name="ck_enrichment_runs_jobs_failed_nonnegative",
        ),
        CheckConstraint(
            "jobs_with_skills >= 0",
            name="ck_enrichment_runs_jobs_with_skills_nonnegative",
        ),
        CheckConstraint(
            "zero_skill_jobs >= 0",
            name="ck_enrichment_runs_zero_skill_jobs_nonnegative",
        ),
        Index("ix_enrichment_runs_status", "status"),
        Index(
            "ix_enrichment_runs_type_versions",
            "run_type",
            "matcher_version",
            "ontology_version",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    matcher_version: Mapped[str] = mapped_column(Text, nullable=False)
    ontology_version: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}", nullable=False)
    jobs_total: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    jobs_processed: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    jobs_succeeded: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    jobs_failed: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    jobs_with_skills: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    zero_skill_jobs: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    job_skills: Mapped[list["JobSkill"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )
    failures: Mapped[list["EnrichmentFailure"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )
    job_results: Mapped[list["JobEnrichmentResult"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class JobSkill(Base):
    """Skill evidence extracted for one job during one enrichment run.

    Offsets are relative to the original field named by `evidence_field`, never to
    the combined matcher input.
    """

    __tablename__ = "job_skills"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "job_id",
            "skill_id",
            "method",
            name="uq_job_skills_run_job_skill_method",
        ),
        CheckConstraint("method IN ('matcher', 'llm')", name="ck_job_skills_method"),
        CheckConstraint(
            "evidence_field IN ('title', 'description')",
            name="ck_job_skills_evidence_field",
        ),
        CheckConstraint("start_char >= 0", name="ck_job_skills_start_char_nonnegative"),
        CheckConstraint("end_char >= start_char", name="ck_job_skills_end_after_start"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_job_skills_confidence_range",
        ),
        Index("ix_job_skills_job_id", "job_id"),
        Index("ix_job_skills_skill_id", "skill_id"),
        Index("ix_job_skills_run_id", "run_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("enrichment_runs.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id"), nullable=False)
    skill_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("skills.id"), nullable=False)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    matched_alias: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_field: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped["EnrichmentRun"] = relationship(back_populates="job_skills")
    job: Mapped["Job"] = relationship(back_populates="job_skills")
    skill: Mapped["Skill"] = relationship(back_populates="job_skills")


class JobEnrichmentResult(Base):
    """Authoritative per-job completion ledger for an enrichment run.

    Positive skill evidence lives in `job_skills`. This table records that each job
    was processed, including successful zero-skill jobs.
    """

    __tablename__ = "job_enrichment_results"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "job_id",
            name="uq_job_enrichment_results_run_job",
        ),
        CheckConstraint(
            "status IN ('success', 'failed')",
            name="ck_job_enrichment_results_status",
        ),
        CheckConstraint(
            "skill_count >= 0",
            name="ck_job_enrichment_results_skill_count_nonnegative",
        ),
        Index("ix_job_enrichment_results_run_id", "run_id"),
        Index("ix_job_enrichment_results_job_id", "job_id"),
        Index("ix_job_enrichment_results_run_status", "run_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("enrichment_runs.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id"), nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    skill_count: Mapped[int] = mapped_column(Integer, nullable=False)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped["EnrichmentRun"] = relationship(back_populates="job_results")
    job: Mapped["Job"] = relationship(back_populates="enrichment_results")

    @validates("status")
    def _validate_status(self, _key: str, value: str) -> str:
        if value not in {"success", "failed"}:
            raise ValueError("status must be 'success' or 'failed'")
        return value

    @validates("skill_count")
    def _validate_skill_count(self, _key: str, value: int) -> int:
        if value < 0:
            raise ValueError("skill_count must be non-negative")
        return value


class EnrichmentFailure(Base):
    """Per-job failure record for retryable/resumable enrichment runs."""

    __tablename__ = "enrichment_failures"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "job_id",
            "stage",
            name="uq_enrichment_failures_run_job_stage",
        ),
        CheckConstraint(
            "attempts >= 1",
            name="ck_enrichment_failures_attempts_positive",
        ),
        Index("ix_enrichment_failures_run_id", "run_id"),
        Index("ix_enrichment_failures_job_id", "job_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("enrichment_runs.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id"), nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, server_default="1", nullable=False)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped["EnrichmentRun"] = relationship(back_populates="failures")
    job: Mapped["Job"] = relationship(back_populates="enrichment_failures")
