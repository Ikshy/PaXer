"""
backend/routers/queue.py
=========================
GET /queue — return the analyst verification queue.

Returns InferenceResults that are pending human sign-off (verified=False),
joined with their parent ImageRecord for context.

Query params
------------
  pending_only : bool  (default True)  — if False, return all results
  limit        : int   (default 50)
  offset       : int   (default 0)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from backend.db.base import get_db
from backend.models.imagery import InferenceResult, ImageRecord
from backend.schemas import QueueItem, QueueResponse

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("", response_model=QueueResponse)
async def get_queue(
    pending_only: bool = Query(default=True, description="Return only unverified results."),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> QueueResponse:
    """
    Return the analyst verification queue.

    Each item is an InferenceResult waiting for human sign-off.
    Sorted by inferred_at ascending (oldest first).
    """
    stmt = (
        select(InferenceResult)
        .options(selectinload(InferenceResult.image))
        .order_by(InferenceResult.inferred_at.asc())
    )

    if pending_only:
        stmt = stmt.where(InferenceResult.verified == False)  # noqa: E712

    # Count total and pending
    from sqlalchemy import func, select as sa_select

    count_stmt = sa_select(func.count()).select_from(InferenceResult)
    pending_stmt = sa_select(func.count()).select_from(InferenceResult).where(
        InferenceResult.verified == False  # noqa: E712
    )

    total_result = await db.execute(count_stmt)
    pending_result = await db.execute(pending_stmt)
    total: int = total_result.scalar_one()
    pending: int = pending_result.scalar_one()

    paged_stmt = stmt.limit(limit).offset(offset)
    rows = await db.execute(paged_stmt)
    inferences = rows.scalars().all()

    items = [
        QueueItem(
            inference_id=inf.id,
            image_id=inf.image_id,
            filename=inf.image.filename if inf.image else "unknown",
            predicted_label=inf.predicted_label,
            confidence=inf.confidence,
            inferred_at=inf.inferred_at,
            verified=inf.verified,
        )
        for inf in inferences
    ]

    return QueueResponse(items=items, total=total, pending=pending)
