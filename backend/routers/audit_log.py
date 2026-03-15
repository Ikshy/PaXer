"""
backend/routers/audit_log.py
=============================
GET /audit — read-only view of the immutable audit log.

Supports filtering by event_type and resource_id.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, select as sa_select

from backend.db.base import get_db
from backend.models.imagery import AuditLog
from backend.schemas import AuditLogEntry, AuditLogResponse

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("", response_model=AuditLogResponse)
async def get_audit_log(
    event_type: str | None = Query(default=None, description="Filter by event type."),
    resource_id: str | None = Query(default=None, description="Filter by resource UUID."),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> AuditLogResponse:
    """
    Retrieve audit log entries (read-only).

    Entries are returned newest-first.  Supports filtering by event_type
    (INGEST, INFER, VERIFY, SYSTEM) and resource_id.
    """
    stmt = select(AuditLog).order_by(AuditLog.timestamp_utc.desc())
    count_stmt = sa_select(func.count()).select_from(AuditLog)

    if event_type:
        stmt = stmt.where(AuditLog.event_type == event_type.upper())
        count_stmt = count_stmt.where(AuditLog.event_type == event_type.upper())
    if resource_id:
        stmt = stmt.where(AuditLog.resource_id == resource_id)
        count_stmt = count_stmt.where(AuditLog.resource_id == resource_id)

    total_result = await db.execute(count_stmt)
    total: int = total_result.scalar_one()

    rows = await db.execute(stmt.limit(limit).offset(offset))
    entries = rows.scalars().all()

    return AuditLogResponse(
        entries=[
            AuditLogEntry(
                id=e.id,
                event_type=e.event_type,
                actor=e.actor,
                resource_type=e.resource_type,
                resource_id=e.resource_id,
                detail=e.detail,
                timestamp_utc=e.timestamp_utc,
            )
            for e in entries
        ],
        total=total,
    )
