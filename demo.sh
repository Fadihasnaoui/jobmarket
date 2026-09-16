#!/bin/sh
# One-command demo: builds and starts Postgres + API + frontend via docker-compose,
# seeding the database from the committed dump (data/seed/jobmarket_seed.dump) on
# first boot. See README.md "Quickstart" for what this does step by step.
set -e

if [ ! -f .env ]; then
    echo "No .env found — copying .env.example. Edit it to add your LLM_API_KEY"
    echo "(required for CV upload/matching features; job data is pre-seeded and"
    echo "needs no API keys) before continuing."
    cp .env.example .env
    echo ""
    echo "Edit .env now, then re-run ./demo.sh"
    exit 0
fi

echo "Building and starting jobmarket (postgres, api, frontend)..."
docker compose up --build -d

echo ""
echo "Waiting for the API to become healthy (first boot restores the seed dataset"
echo "and can take a couple of minutes)..."

deadline=$(($(date +%s) + 300))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
    status=$(docker inspect --format='{{.State.Health.Status}}' jobmarket-api 2>/dev/null || echo "")
    if [ "$status" = "healthy" ]; then
        healthy=1
        break
    fi
    sleep 5
done

echo ""
if [ "$healthy" = "1" ]; then
    echo "jobmarket is up."
else
    echo "Still starting (or something went wrong) — check with:"
    echo "  docker compose logs -f api"
fi
echo ""
echo "  Frontend:  http://localhost:5173"
echo "  API docs:  http://localhost:8123/docs"
echo ""
echo "Stop everything with: docker compose down"
