"""Uvicorn entrypoint: `uvicorn jobmarket.api.main:app --reload`."""

from __future__ import annotations

from jobmarket.api.app import create_app

app = create_app()
