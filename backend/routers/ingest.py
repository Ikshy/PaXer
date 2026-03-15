"""
backend/routers/ingest.py
==========================
POST /ingest — upload a synthetic satellite image.

Accepts a multipart file upload, validates it is a real image, persists
it to the local object store, creates an ImageRecord in the DB, and writes
an INGEST audit log entry.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.base import get_db
from backend.models.imagery import ImageRecord
from backend.schemas import IngestResponse
from backend.services import audit, store

router = APIRouter(prefix="/ingest", tags=["ingest"])

_ALLOWED_MIME = {"image/png", "image/jpeg", "image/tiff"}
_MAX_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_image(
    file: UploadFile = File(..., description="Synthetic satellite image (PNG/JPEG/TIFF)."),
    actor: str = Form(default="anonymous", description="Operator ID for audit log."),
    db: AsyncSession = Depends(get_db),
) -> IngestResponse:
    """
    Ingest a synthetic satellite image.

    - Validates MIME type and file size.
    - Stores image in local object store.
    - Creates an ImageRecord row in the database.
    - Writes an immutable INGEST audit log entry.

    **Safety note**: only synthetic or public imagery should be uploaded.
    """
    # --- Validate MIME ---
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type {file.content_type!r}. "
                   f"Allowed: {sorted(_ALLOWED_MIME)}",
        )

    # --- Read bytes ---
    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(data)} bytes). Maximum is {_MAX_BYTES} bytes.",
        )
    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --- Persist to object store ---
    image_id = str(uuid.uuid4())
    filename = file.filename or f"{image_id}.png"

    try:
        store_path, sha256, width, height = store.save_upload(image_id, filename, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image decode failed: {exc}",
        ) from exc

    # --- Persist metadata to DB ---
    record = ImageRecord(
        id=image_id,
        filename=filename,
        store_path=store_path,
        sha256=sha256,
        width=width,
        height=height,
        is_synthetic=True,
        ingested_by=actor,
    )
    db.add(record)

    # --- Audit log ---
    await audit.log_event(
        db,
        event_type=audit.EVENT_INGEST,
        actor=actor,
        resource_type=audit.RESOURCE_IMAGE,
        resource_id=image_id,
        detail={
            "filename": filename,
            "sha256": sha256,
            "width": width,
            "height": height,
            "bytes": len(data),
        },
    )

    await db.commit()
    await db.refresh(record)

    return IngestResponse(
        image_id=record.id,
        filename=record.filename,
        sha256=record.sha256,
        width=record.width,
        height=record.height,
        is_synthetic=record.is_synthetic,
        store_path=record.store_path,
        ingested_at=record.ingested_at,
    )
