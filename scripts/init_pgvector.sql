-- Run once as a PostgreSQL superuser (e.g. postgres).
-- Required only when you need vector search later; safe to skip for Step 1.
-- Docker Compose runs this automatically on first container start.

CREATE EXTENSION IF NOT EXISTS vector;
