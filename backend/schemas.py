"""
backend/schemas.py
==================
Pydantic v2 schemas for all API request and response bodies.
Kept in one file for Part 3; split by domain in later parts if needed.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    environment: str


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """Returned after a successful image upload."""

    image_id: str
    filename: str
    sha256: str
    width: int
    height: int
    is_synthetic: bool
    store_path: str
    ingested_at: datetime
    message: str = "Image ingested. Ready for inference."


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class InferRequest(BaseModel):
    """Body for POST /infer — specifies which image to run inference on."""

    image_id: str = Field(..., description="UUID of a previously ingested ImageRecord.")
    analyst_id: str = Field(
        default="anonymous",
        description="Identifier of the operator requesting inference.",
    )


class InferResponse(BaseModel):
    """Full inference result, including provenance and verification status."""

    inference_id: str
    image_id: str
    predicted_class: int
    predicted_label: str
    confidence: float
    logits: list[float]
    model_arch: str
    model_checkpoint: str
    model_sha256: str
    verified: bool
    inferred_at: datetime
    safety_note: str = (
        "This result is a model candidate only. "
        "Human analyst sign-off via /verify is required before it is authoritative."
    )


# ---------------------------------------------------------------------------
# Analyst queue
# ---------------------------------------------------------------------------


class QueueItem(BaseModel):
    """A single item in the analyst verification queue."""

    inference_id: str
    image_id: str
    filename: str
    predicted_label: str
    confidence: float
    inferred_at: datetime
    verified: bool


class QueueResponse(BaseModel):
    items: list[QueueItem]
    total: int
    pending: int


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    """Body for POST /verify — human analyst sign-off on an inference result."""

    inference_id: str = Field(..., description="UUID of the InferenceResult to verify.")
    analyst_id: str = Field(..., description="Identifier of the human analyst.")
    confirmed: bool = Field(
        ...,
        description="True if analyst confirms the model prediction; False to reject.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Free-text analyst notes (required when confirmed=False).",
    )

    @field_validator("notes")
    @classmethod
    def notes_required_on_rejection(cls, v: Optional[str], info: Any) -> Optional[str]:
        data = info.data
        if data.get("confirmed") is False and not v:
            raise ValueError("notes are required when confirmed=False")
        return v


class VerifyResponse(BaseModel):
    inference_id: str
    verified: bool
    verified_by: str
    verified_at: datetime
    analyst_notes: Optional[str]
    message: str


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class AuditLogEntry(BaseModel):
    id: int
    event_type: str
    actor: str
    resource_type: str
    resource_id: str
    detail: str
    timestamp_utc: datetime


class AuditLogResponse(BaseModel):
    entries: list[AuditLogEntry]
    total: int
