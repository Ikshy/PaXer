"""
backend/db/base.py
==================
SQLAlchemy async engine and session factory.

In test mode (TESTING=1 env var) an in-memory SQLite database is used so
tests never require a running Postgres instance.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


# ---------------------------------------------------------------------------
# Declarative base shared by all ORM models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------


def _make_engine():  # type: ignore[return]
    """Return an async engine; SQLite (aiosqlite) in test mode, Postgres otherwise."""
    if os.getenv("TESTING", "0") == "1":
        # In-memory SQLite for unit/integration tests — no Postgres needed
        return create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
        )

    from backend.config import get_settings  # local import avoids circular

    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=os.getenv("APP_ENV", "development") == "development",
        pool_pre_ping=True,
    )


engine = _make_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session; closes on exit."""
    async with AsyncSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Schema initialisation helper (used in tests and dev startup)
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
