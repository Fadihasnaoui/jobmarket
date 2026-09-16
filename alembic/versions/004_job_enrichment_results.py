"""Add per-job enrichment completion ledger."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "004_job_enrichment_results"
down_revision: str | None = "003_enrichment_layer"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "job_enrichment_results",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("skill_count", sa.Integer(), nullable=False),
        sa.Column(
            "processed_at",
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
            "status IN ('success', 'failed')",
            name="ck_job_enrichment_results_status",
        ),
        sa.CheckConstraint(
            "skill_count >= 0",
            name="ck_job_enrichment_results_skill_count_nonnegative",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["enrichment_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "job_id",
            name="uq_job_enrichment_results_run_job",
        ),
    )
    op.create_index(
        "ix_job_enrichment_results_job_id",
        "job_enrichment_results",
        ["job_id"],
    )
    op.create_index(
        "ix_job_enrichment_results_run_id",
        "job_enrichment_results",
        ["run_id"],
    )
    op.create_index(
        "ix_job_enrichment_results_run_status",
        "job_enrichment_results",
        ["run_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_job_enrichment_results_run_status",
        table_name="job_enrichment_results",
    )
    op.drop_index("ix_job_enrichment_results_run_id", table_name="job_enrichment_results")
    op.drop_index("ix_job_enrichment_results_job_id", table_name="job_enrichment_results")
    op.drop_table("job_enrichment_results")
