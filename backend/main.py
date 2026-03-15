"""
backend/main.py
===============
FastAPI application factory for CTHMP backend.

Mounts all routers, configures CORS, and initialises the DB schema on
startup (idempotent — safe to run against an existing schema).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.db.base import init_db
from backend.routers import audit_log, infer, ingest, queue, verify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise DB tables on startup."""
    logger.info("CTHMP backend starting — initialising database schema…")
    await init_db()
    logger.info("Database ready.")
    yield
    logger.info("CTHMP backend shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="CTHMP API",
        description=(
            "Conflict Transparency & Humanitarian Monitoring Platform — "
            "research-grade, humanitarian use only."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # --- CORS (restrict in production) ---
    origins = ["http://localhost:5173", "http://localhost:3000"]
    if settings.app_env == "development":
        origins.append("*")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routers ---
    app.include_router(ingest.router)
    app.include_router(infer.router)
    app.include_router(queue.router)
    app.include_router(verify.router)
    app.include_router(audit_log.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        from backend.schemas import HealthResponse
        return HealthResponse(
            status="ok",
            environment=settings.app_env,
        ).model_dump()

    return app


app = create_app()
