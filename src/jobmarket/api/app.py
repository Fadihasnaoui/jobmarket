"""FastAPI application factory for the jobmarket API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from jobmarket.api import (
    chat_routes,
    cv_routes,
    improve_routes,
    quality_routes,
    stats_routes,
)


class HealthResponse(BaseModel):
    status: str = "ok"

API_TITLE = "jobmarket API"
API_DESCRIPTION = (
    "CV parsing, extraction, embedding, and job matching — the same validated "
    "pipeline the CLI uses, over HTTP."
)
API_VERSION = "0.1.0"


def create_app() -> FastAPI:
    app = FastAPI(title=API_TITLE, description=API_DESCRIPTION, version=API_VERSION)

    # Permissive by design: this API is meant to be driven by a local frontend during
    # development/demoing, not deployed publicly behind this config as-is.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_routes.router)
    app.include_router(cv_routes.router)
    app.include_router(improve_routes.router)
    app.include_router(quality_routes.router)
    app.include_router(stats_routes.router)

    @app.get("/health", tags=["meta"], response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    return app


__all__ = ["create_app"]
