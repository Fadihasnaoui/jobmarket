# jobmarket

A job-market intelligence pipeline: ingest real job postings from multiple boards,
enrich them with a deterministic skills ontology and semantic embeddings, then match a
candidate's CV against that corpus — with a scored quality report, a market skill-gap
analysis, and a downloadable, grounded CV rewrite. FastAPI backend, React frontend,
PostgreSQL + pgvector.

The one rule that shapes every layer: **never fabricate**. Every skill, match
explanation, or generated sentence traces back to real source text or a real database
number — see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for exactly where that
guarantee does and doesn't reach today.

## Quickstart (Docker)

```powershell
git clone <repo-url> jobmarket
cd jobmarket
.\demo.ps1        # PowerShell — or ./demo.sh on macOS/Linux
```

First run copies `.env.example` to `.env` and asks you to add an `LLM_API_KEY` (any
OpenAI-compatible provider — Groq, OpenAI, or a local Ollama server all work, see
[Environment variables](#environment-variables)). Re-run the script once that's set.

`docker compose up --build` (what the script wraps) then:

1. Starts Postgres (`pgvector/pgvector:pg16`).
2. Builds and starts the API, which on first boot runs migrations, syncs the skills
   ontology, and — since a fresh database has no jobs — restores the committed seed
   dataset (`data/seed/jobmarket_seed.dump`: ~36k real, enriched job postings with
   skills and embeddings already computed, `raw_jobs` excluded to keep it small).
   No Adzuna/Jooble API keys are needed for this — only for ingesting *new* jobs later.
3. Builds and starts the frontend, served on port 5173.

```
Frontend:  http://localhost:5173
API docs:  http://localhost:8123/docs
```

Stop everything with `docker compose down` (add `-v` to also drop the Postgres volume
and start fresh next time).

## Architecture

```mermaid
flowchart TD
    subgraph Sources["Job boards"]
        Adzuna[Adzuna]
        RemoteOK[RemoteOK]
        Jooble[Jooble]
    end

    Adzuna --> Ingest
    RemoteOK --> Ingest
    Jooble --> Ingest
    Ingest["Ingest\n(verbatim, append-only)"] --> RawJobs[(raw_jobs)]
    RawJobs --> Parse["Parse + normalize + dedup"]
    Parse --> Jobs[(jobs, companies)]

    Jobs --> Matcher["Deterministic skills matcher\n(ontology alias scan)"]
    Matcher --> JobSkills[(job_skills)]
    Jobs --> Embed["Embed\n(sentence-transformers, CPU)"]
    Embed --> JobsEmbed[("jobs.embedding\n(pgvector)")]

    subgraph CV["CV pipeline (per upload, no persistence)"]
        Upload[Upload PDF/DOCX/TXT] --> ParseCV[Parse document]
        ParseCV --> Extract["Extract\n(one-shot LLM, deterministic fallback)"]
        Extract --> Guard["Grounding guard\n(evidence + offset + ontology check)"]
        Guard --> Profile[CvProfile]
    end

    Profile --> Match["Matching\n(lexical / semantic / hybrid)"]
    JobSkills --> Match
    JobsEmbed --> Match
    Match --> Recs[Ranked recommendations]

    Recs --> Gap["Skill-gap analysis\n(real corpus demand)"]
    Recs --> Improve["CV improver\n(grounded rewrite, LLM + guard)"]
    Profile --> QReport["Quality report\n(6 scored checks)"]
    Improve --> QReport

    Recs --> API
    Gap --> API
    Improve --> API
    QReport --> API
    Profile --> API
    API["FastAPI"] --> Frontend["React frontend"]
```

Two hard boundaries worth knowing before reading the code:

- **Raw is append-only, truth lives in the parsed tables.** A parser bug gets fixed by
  re-parsing from `raw_jobs`, never by re-scraping.
- **Nothing an LLM generates ships unchecked.** Skills need an ontology match plus an
  exact, offset-verified quote from the source text; generated CV summaries/bullets are
  scanned against the candidate's own grounded facts and rejected (falling back to the
  original wording) if they introduce anything that isn't already real. See
  `docs/LIMITATIONS.md` for the one place this is closer to "logged" than "enforced"
  (prompt injection on ungrounded fields) and the one place the CV path's guard doesn't
  currently combine with the deterministic matcher.

## Manual setup (no Docker)

Useful for iterating on backend/frontend code directly instead of rebuilding images.

### Backend

```powershell
cd jobmarket
copy .env.example .env
pip install -e ".[dev]"
python -m alembic upgrade head
python -m jobmarket skills sync
uvicorn jobmarket.api.app:create_app --factory --reload --port 8123
```

Postgres needs to be reachable at whatever `DATABASE_URL`/`DATABASE_URL_SYNC` point to
— either `docker compose up -d postgres` (just the DB service) or a local install; see
[Environment variables](#environment-variables). To load the same seed data the demo
uses:

```powershell
$env:PGPASSWORD = "jobmarket"
& "path\to\pg_restore.exe" --data-only --disable-triggers --no-owner `
    -d "postgresql://jobmarket:jobmarket@localhost:5432/jobmarket" `
    data\seed\jobmarket_seed.dump
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Defaults to `VITE_API_BASE_URL=http://127.0.0.1:8123` (see `frontend/src/api/client.ts`);
override via `frontend/.env.local` if the API runs elsewhere.

## Environment variables

| Variable | Required for | Description |
|---|---|---|
| `DATABASE_URL` | Always | Async SQLAlchemy URL (`postgresql+asyncpg://...`) |
| `DATABASE_URL_SYNC` | Alembic, seed restore | Sync URL (`postgresql+psycopg://...`) |
| `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` | Ingesting new Adzuna jobs | From [developer.adzuna.com](https://developer.adzuna.com/) — not needed for the seeded demo |
| `JOOBLE_API_KEY` | Ingesting new Jooble jobs | From [jooble.org/api/about](https://jooble.org/api/about) — not needed for the seeded demo |
| `LLM_PROVIDER` / `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | CV upload, improve, quality report | Any OpenAI-compatible chat-completions endpoint — Groq, OpenAI, a local Ollama server, etc. Required for every CV feature; the seeded job corpus works without it. |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` / `EMBEDDING_BATCH_SIZE` / `EMBEDDING_NORMALIZE` | Semantic/hybrid matching | Defaults to a multilingual sentence-transformers model, baked into the Docker image at build time |

RemoteOK requires no API key but needs a realistic User-Agent (handled automatically).

## CLI reference

Core pipeline:

```powershell
python -m jobmarket ingest --source adzuna   # or remoteok / jooble / --all
python -m jobmarket parse                     # or --source adzuna [--reparse]
python -m jobmarket stats
python -m jobmarket dedup-report

python -m jobmarket skills sync                        # upsert data/skills_ontology.yaml
python -m jobmarket enrichment run-matcher --yes        # deterministic skill matching
python -m jobmarket embed jobs                          # compute + persist job embeddings

python -m jobmarket cv match resume.pdf                  # rank jobs against one CV, from the terminal
python -m jobmarket cv recommend --cv resume.pdf --mode hybrid
```

Re-running `ingest` is safe (`ON CONFLICT DO NOTHING`); re-running `skills sync` and
`enrichment run-matcher` is safe (upsert semantics).

Additional internal tooling — CV-extraction gold-dataset preparation/validation
(`jobmarket dataset ...`), extraction accuracy evaluation (`jobmarket evaluate ...`),
and the offline LLM cost/benefit benchmark (`jobmarket llm benchmark`) — exists for
this project's own development; run `python -m jobmarket --help` (or `<command>
--help`) to see it.

## API reference

All routes are under the FastAPI app in `src/jobmarket/api/`; interactive docs at
`/docs` once running.

| Method & path | Purpose |
|---|---|
| `POST /cv/upload` | Parse + extract + embed one CV; returns a short-lived `cv_id` (30 min TTL, in-memory only — raw CV text is never persisted) |
| `GET /cv/{cv_id}/matches` | Ranked job recommendations (lexical / semantic / hybrid) |
| `GET /cv/{cv_id}/skill-gap` | Missing skills across top matches, ranked by real corpus demand |
| `POST /cv/{cv_id}/improve` | Grounded CV rewrite + single-column `.docx` download |
| `POST /cv/{cv_id}/quality-report` | Six-category scored diagnostic + two-column `.docx` download |
| `GET /stats/overview` | Corpus-wide totals and coverage |
| `GET /stats/skills` | Skill demand across the corpus, optionally filtered |
| `GET /health` | Liveness check |

## Frontend

React + Vite + Tailwind, no server-side rendering. Tabs (`frontend/src/pages/`):

- **Upload** — drag-and-drop a CV, see the extracted profile
- **Matches** — ranked jobs with a "why this score" breakdown, plus the CV-improve panel
- **Skill Gap** — missing skills ranked by market demand
- **Quality Report** — score ring, per-category pass/warning/fail sidebar, detail pane, `.docx` download
- **Dashboard** — corpus-wide charts (not CV-scoped)

`frontend/src/context/CvSessionContext.tsx` holds the one shared piece of state
(`cv_id`) across tabs.

## Testing

```powershell
python -m pytest tests\unit -q
python -m pytest tests\integration -q
python -m ruff check src tests
python -m mypy src\jobmarket

cd frontend
npx tsc --noEmit
npm run build
```

All HTTP/LLM calls in tests are mocked — no network access or API keys required to run
the suite. Integration tests need a reachable Postgres (same `DATABASE_URL` as normal
use); they create and clean up their own rows, never touching `raw_jobs`/`jobs` data
from a real ingest.

## Project layout

```
jobmarket/
├── docker-compose.yml       # postgres + api + frontend
├── Dockerfile               # backend image
├── demo.ps1 / demo.sh       # one-command docker-compose demo
├── data/
│   ├── skills_ontology.yaml
│   └── seed/jobmarket_seed.dump   # committed demo dataset (data-only, no raw_jobs)
├── alembic/                 # schema migrations
├── src/jobmarket/
│   ├── config.py, cli.py
│   ├── db/                  # SQLAlchemy models, async session
│   ├── ingest/               # AbstractSource, sources (Adzuna/RemoteOK/Jooble), runner
│   ├── parse/                 # AbstractParser, normalize, dedup, runner
│   ├── skills/                # ontology loader, deterministic matcher, LLM extractor
│   ├── embeddings/            # sentence-transformers job/CV encoders
│   ├── cv/                    # parsing, LLM extraction + grounding guard, matching,
│   │                          # improver, quality report, docx export
│   ├── api/                   # FastAPI routes + Pydantic models
│   └── reporting/              # stats, dedup-report
├── frontend/                 # React + Vite + Tailwind
├── docs/LIMITATIONS.md       # known trust-boundary gaps, by design not bugs
??? archive/                  # recoverable historical audits and retired development artefacts
└── tests/{unit,integration}
```

## Explicit non-goals

- User accounts / authentication / multi-tenancy
- Persisting uploaded CV text or contact info anywhere (by design — see `cv/profile.py`
  and `api/store.py`)
- Production hardening of the API's CORS/security config — permissive by design for
  local demo use (`api/app.py`)
- LinkedIn / Indeed scraping
- Horizontal scaling / job queues for ingestion or enrichment
