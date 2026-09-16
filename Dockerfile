# Backend image: FastAPI app + CLI, editable install so `data/skills_ontology.yaml`
# resolves the same way it does in local dev (jobmarket.skills.ontology derives its
# path from `__file__`, three parents up from src/jobmarket/skills/ — see that module's
# docstring). A non-editable install would relocate the package into site-packages and
# break that path, so this intentionally does NOT use a build/wheel step.
FROM python:3.11-slim

# postgresql-client: pg_restore/psql, used only by docker-entrypoint.sh to seed the
# database on first boot. build-essential: safety net for any dependency without a
# manylinux wheel on this platform; everything currently pinned does ship one, but the
# cost of keeping this is a slightly larger image, not a broken build if that changes.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./
COPY data ./data
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh

# CPU-only torch: sentence-transformers pulls the default (CUDA-capable, multi-GB) torch
# wheel otherwise; this project only ever runs embeddings on CPU (see EMBEDDING_* env
# vars — no CUDA/device selection exists anywhere in the pipeline).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

# Bake the sentence-transformers model into the image so a fresh container doesn't need
# a live Hugging Face download on its first CV upload — same model name as
# .env.example's EMBEDDING_MODEL default; override that env var and rebuild if it
# changes, otherwise this cached copy is what actually loads at runtime regardless.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

RUN chmod +x ./scripts/docker-entrypoint.sh

EXPOSE 8123

ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
