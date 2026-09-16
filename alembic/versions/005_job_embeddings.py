"""Add pgvector job embedding storage."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.types import UserDefinedType

revision: str = "005_job_embeddings"
down_revision: str | None = "004_job_enrichment_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


class Vector(UserDefinedType[list[float]]):
    """Minimal pgvector type for Alembic DDL."""

    cache_ok = True

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def get_col_spec(self, **_kw: object) -> str:
        return f"vector({self.dimension})"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("jobs", sa.Column("embedding", Vector(384), nullable=True))
    op.add_column("jobs", sa.Column("embedding_model", sa.Text(), nullable=True))
    op.add_column("jobs", sa.Column("embedding_version", sa.Text(), nullable=True))
    op.add_column(
        "jobs",
        sa.Column("embedding_created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_jobs_embedding_model_version",
        "jobs",
        ["embedding_model", "embedding_version"],
    )
    op.execute(
        "CREATE INDEX ix_jobs_embedding_hnsw_cosine "
        "ON jobs USING hnsw (embedding vector_cosine_ops) "
        "WHERE embedding IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_jobs_embedding_hnsw_cosine")
    op.drop_index("ix_jobs_embedding_model_version", table_name="jobs")
    op.drop_column("jobs", "embedding_created_at")
    op.drop_column("jobs", "embedding_version")
    op.drop_column("jobs", "embedding_model")
    op.drop_column("jobs", "embedding")
