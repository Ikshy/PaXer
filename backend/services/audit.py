"""
backend/services/audit.py
==========================
Service for writing immutable audit log entries.

Every INGEST, INFER, and VERIFY event must pass through this service.
Rows are never updated or deleted — only appended.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.imagery import AuditLog


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

EVENT_INGEST = "INGEST"
EVENT_INFER = "INFER"
EVENT_VERIFY = "VERIFY"
EVENT_SYSTEM = "SYSTEM"

RESOURCE_IMAGE = "ImageRecord"
RESOURCE_INFERENCE = "InferenceResult"


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------


async def log_event(
    db: AsyncSession,
    *,
    event_type: str,
    actor: str,
    resource_type: str,
    resource_id: str,
    detail: dict[str, Any] | None = None,
) -> AuditLog:
    """
    Append one immutable audit log entry.

    Parameters
    ----------
    db            : active async session (caller is responsible for commit)
    event_type    : one of EVENT_* constants
    actor         : human or system identifier
    resource_type : RESOURCE_* constant
    resource_id   : UUID of the affected resource
    detail        : arbitrary JSON-serialisable metadata

    Returns
    -------
    The newly created (unsaved) AuditLog ORM object.
    The caller's transaction will persist it.
    """
    entry = AuditLog(
        event_type=event_type,
        actor=actor,
        resource_type=resource_type,
        resource_id=resource_id,
        detail=json.dumps(detail or {}),
    )
    db.add(entry)
    return entry
