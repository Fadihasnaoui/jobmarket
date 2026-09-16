#!/bin/sh
# Backend container startup: migrate, sync the skills ontology, seed demo data on an
# empty database, then serve. Every step here is safe to run again on container
# restart — migrations are idempotent by design, `skills sync` upserts, and the seed
# restore only fires when `jobs` is genuinely empty.
set -e

echo "[entrypoint] running migrations..."
alembic upgrade head

echo "[entrypoint] syncing skills ontology..."
python -m jobmarket skills sync

# pg_restore/psql want a plain libpq URI; DATABASE_URL_SYNC uses SQLAlchemy's
# "+psycopg" driver suffix, which those tools don't understand.
PG_URI=$(echo "$DATABASE_URL_SYNC" | sed 's#postgresql+psycopg://#postgresql://#')
SEED_FILE="/app/data/seed/jobmarket_seed.dump"

JOB_COUNT=$(psql "$PG_URI" -tAc "SELECT count(*) FROM jobs" 2>/dev/null || echo 0)

if [ "$JOB_COUNT" = "0" ] && [ -f "$SEED_FILE" ]; then
    echo "[entrypoint] jobs table is empty and a seed dump exists — restoring demo data..."
    pg_restore --data-only --disable-triggers --no-owner -d "$PG_URI" "$SEED_FILE"
    echo "[entrypoint] seed restore complete."
elif [ "$JOB_COUNT" = "0" ]; then
    echo "[entrypoint] jobs table is empty and no seed dump was found at $SEED_FILE —"
    echo "[entrypoint] starting with no job data. Run 'jobmarket ingest --all' and"
    echo "[entrypoint] 'jobmarket enrichment run-matcher' to populate it, or see the README."
else
    echo "[entrypoint] jobs table already has $JOB_COUNT rows — skipping seed restore."
fi

echo "[entrypoint] starting API server..."
exec uvicorn jobmarket.api.app:create_app --factory --host 0.0.0.0 --port 8123
