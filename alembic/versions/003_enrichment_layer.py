"""Enrichment layer persistence tables."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003_enrichment_layer"
down_revision: str | None = "002_skills_ontology"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "enrichment_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_type", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("matcher_version", sa.Text(), nullable=False),
        sa.Column("ontology_version", sa.Text(), nullable=False),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("jobs_total", sa.Integer(), server_default="0", nullable=False),
        sa.Column("jobs_processed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("jobs_succeeded", sa.Integer(), server_default="0", nullable=False),
        sa.Column("jobs_failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("jobs_with_skills", sa.Integer(), server_default="0", nullable=False),
        sa.Column("zero_skill_jobs", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "run_type IN ('matcher', 'llm_experiment')",
            name="ck_enrichment_runs_run_type",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'success', 'partial', 'failed')",
            name="ck_enrichment_runs_status",
        ),
        sa.CheckConstraint(
            "jobs_total >= 0",
            name="ck_enrichment_runs_jobs_total_nonnegative",
        ),
        sa.CheckConstraint(
            "jobs_processed >= 0",
            name="ck_enrichment_runs_jobs_processed_nonnegative",
        ),
        sa.CheckConstraint(
            "jobs_succeeded >= 0",
            name="ck_enrichment_runs_jobs_succeeded_nonnegative",
        ),
        sa.CheckConstraint(
            "jobs_failed >= 0",
            name="ck_enrichment_runs_jobs_failed_nonnegative",
        ),
        sa.CheckConstraint(
            "jobs_with_skills >= 0",
            name="ck_enrichment_runs_jobs_with_skills_nonnegative",
        ),
        sa.CheckConstraint(
            "zero_skill_jobs >= 0",
            name="ck_enrichment_runs_zero_skill_jobs_nonnegative",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_enrichment_runs_status", "enrichment_runs", ["status"])
    op.create_index(
        "ix_enrichment_runs_type_versions",
        "enrichment_runs",
        ["run_type", "matcher_version", "ontology_version"],
    )

    op.create_table(
        "job_skills",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.BigInteger(), nullable=False),
        sa.Column("skill_id", sa.BigInteger(), nullable=False),
        sa.Column("method", sa.Text(), nullable=False),
        sa.Column("matched_alias", sa.Text(), nullable=True),
        sa.Column("evidence_text", sa.Text(), nullable=True),
        sa.Column("evidence_field", sa.Text(), nullable=False),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("method IN ('matcher', 'llm')", name="ck_job_skills_method"),
        sa.CheckConstraint(
            "evidence_field IN ('title', 'description')",
            name="ck_job_skills_evidence_field",
        ),
        sa.CheckConstraint(
            "start_char >= 0",
            name="ck_job_skills_start_char_nonnegative",
        ),
        sa.CheckConstraint(
            "end_char >= start_char",
            name="ck_job_skills_end_after_start",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="ck_job_skills_confidence_range",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["enrichment_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["skill_id"], ["skills.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "job_id",
            "skill_id",
            "method",
            name="uq_job_skills_run_job_skill_method",
        ),
    )
    op.create_index("ix_job_skills_job_id", "job_skills", ["job_id"])
    op.create_index("ix_job_skills_run_id", "job_skills", ["run_id"])
    op.create_index("ix_job_skills_skill_id", "job_skills", ["skill_id"])

    op.create_table(
        "enrichment_failures",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.BigInteger(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="1", nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "attempts >= 1",
            name="ck_enrichment_failures_attempts_positive",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["enrichment_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "job_id",
            "stage",
            name="uq_enrichment_failures_run_job_stage",
        ),
    )
    op.create_index("ix_enrichment_failures_job_id", "enrichment_failures", ["job_id"])
    op.create_index("ix_enrichment_failures_run_id", "enrichment_failures", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_enrichment_failures_run_id", table_name="enrichment_failures")
    op.drop_index("ix_enrichment_failures_job_id", table_name="enrichment_failures")
    op.drop_table("enrichment_failures")

    op.drop_index("ix_job_skills_skill_id", table_name="job_skills")
    op.drop_index("ix_job_skills_run_id", table_name="job_skills")
    op.drop_index("ix_job_skills_job_id", table_name="job_skills")
    op.drop_table("job_skills")

    op.drop_index("ix_enrichment_runs_type_versions", table_name="enrichment_runs")
    op.drop_index("ix_enrichment_runs_status", table_name="enrichment_runs")
    op.drop_table("enrichment_runs")
