"""
backend/routers/infer.py
=========================
POST /infer — run the scene classifier on an ingested image.

Loads image bytes from the object store, calls the inference service,
persists an InferenceResult, and writes an INFER audit log entry.

The response includes a prominent safety_note reminding callers that
human analyst verification is required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.db.base import get_db
from backend.models.imagery import ImageRecord, InferenceResult
from backend.schemas import InferRequest, InferResponse
from backend.services import audit, store
from backend.services.inference import run_inference

router = APIRouter(prefix="/infer", tags=["infer"])


@router.post("", response_model=InferResponse, status_code=status.HTTP_200_OK)
async def infer_image(
    body: InferRequest,
    db: AsyncSession = Depends(get_db),
) -> InferResponse:
    """
    Run model inference on a previously ingested image.

    - Fetches the ImageRecord and loads image bytes from store.
    - Runs the SceneClassifier forward pass.
    - Persists an InferenceResult (verified=False by default).
    - Writes an immutable INFER audit log entry.

    **Result is a candidate only** — human analyst sign-off via /verify required.
    """
    # --- Fetch image record ---
    result = await db.execute(select(ImageRecord).where(ImageRecord.id == body.image_id))
    record: ImageRecord | None = result.scalar_one_or_none()

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ImageRecord {body.image_id!r} not found.",
        )

    # --- Load image bytes from store ---
    try:
        image_bytes = store.load_image_bytes(record.store_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found in store: {exc}",
        ) from exc

    # --- Run inference ---
    try:
        pred = run_inference(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    # --- Persist inference result ---
    inference_id = str(uuid.uuid4())
    inference = InferenceResult(
        id=inference_id,
        image_id=record.id,
        predicted_class=pred["predicted_class"],
        predicted_label=pred["predicted_label"],
        confidence=pred["confidence"],
        logits_json=pred["logits_json"],
        model_arch=pred["model_arch"],
        model_checkpoint=pred["model_checkpoint"],
        model_sha256=pred["model_sha256"],
        verified=False,
    )
    db.add(inference)

    # --- Audit log ---
    await audit.log_event(
        db,
        event_type=audit.EVENT_INFER,
        actor=body.analyst_id,
        resource_type=audit.RESOURCE_INFERENCE,
        resource_id=inference_id,
        detail={
            "image_id": record.id,
            "predicted_label": pred["predicted_label"],
            "confidence": pred["confidence"],
            "model_sha256": pred["model_sha256"][:12],
        },
    )

    await db.commit()
    await db.refresh(inference)

    import json as _json

    return InferResponse(
        inference_id=inference.id,
        image_id=inference.image_id,
        predicted_class=inference.predicted_class,
        predicted_label=inference.predicted_label,
        confidence=inference.confidence,
        logits=_json.loads(inference.logits_json),
        model_arch=inference.model_arch,
        model_checkpoint=inference.model_checkpoint,
        model_sha256=inference.model_sha256,
        verified=inference.verified,
        inferred_at=inference.inferred_at,
    )
