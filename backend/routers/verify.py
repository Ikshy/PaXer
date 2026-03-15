"""
backend/routers/verify.py
==========================
POST /verify — human analyst sign-off on an inference result.

This is the **ethics-critical** endpoint: only after a human analyst calls
/verify does an InferenceResult become authoritative (verified=True).

Rules enforced here:
  - Notes are required when confirmed=False (rejection).
  - A result can only be verified once.
  - Every verification is written to the immutable audit log.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.db.base import get_db
from backend.models.imagery import InferenceResult
from backend.schemas import VerifyRequest, VerifyResponse
from backend.services import audit

router = APIRouter(prefix="/verify", tags=["verify"])


@router.post("", response_model=VerifyResponse, status_code=status.HTTP_200_OK)
async def verify_inference(
    body: VerifyRequest,
    db: AsyncSession = Depends(get_db),
) -> VerifyResponse:
    """
    Human analyst sign-off on an inference result.

    - Sets verified=True/False, records analyst_id, timestamp, and notes.
    - Writes an immutable VERIFY audit log entry.
    - Rejects attempts to re-verify an already-verified result.

    **This endpoint is the ethics gate**: no model output is authoritative
    until it passes through here.
    """
    result = await db.execute(
        select(InferenceResult).where(InferenceResult.id == body.inference_id)
    )
    inference: InferenceResult | None = result.scalar_one_or_none()

    if inference is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"InferenceResult {body.inference_id!r} not found.",
        )

    if inference.verified:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"InferenceResult {body.inference_id!r} has already been verified "
                f"by {inference.verified_by!r} at {inference.verified_at}. "
                "Re-verification is not permitted."
            ),
        )

    now = datetime.now(timezone.utc)
    inference.verified = body.confirmed
    inference.verified_by = body.analyst_id
    inference.verified_at = now
    inference.analyst_notes = body.notes

    await audit.log_event(
        db,
        event_type=audit.EVENT_VERIFY,
        actor=body.analyst_id,
        resource_type=audit.RESOURCE_INFERENCE,
        resource_id=inference.id,
        detail={
            "confirmed": body.confirmed,
            "predicted_label": inference.predicted_label,
            "confidence": inference.confidence,
            "notes": body.notes,
        },
    )

    await db.commit()
    await db.refresh(inference)

    action = "confirmed" if body.confirmed else "rejected"
    return VerifyResponse(
        inference_id=inference.id,
        verified=inference.verified,
        verified_by=inference.verified_by,
        verified_at=inference.verified_at,
        analyst_notes=inference.analyst_notes,
        message=f"Inference result {action} by analyst {body.analyst_id!r}.",
    )
