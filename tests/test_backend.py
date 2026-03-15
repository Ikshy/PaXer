"""
tests/test_backend.py
======================
Integration tests for the CTHMP FastAPI backend.

Uses httpx.AsyncClient + pytest-anyio so tests run fully in-process
against a SQLite in-memory database (TESTING=1 env var switches the engine).

No running Postgres, no MinIO, no real model checkpoint required:
  - DB is SQLite in-memory (via TESTING=1)
  - Object store writes to a temp directory (LOCAL_STORE_DIR env var)
  - Model inference is mocked so tests never need a .pt file

Test coverage:
  - GET  /health
  - POST /ingest  (happy path, bad MIME, empty file)
  - POST /infer   (happy path, unknown image_id)
  - GET  /queue   (pending_only, all)
  - POST /verify  (confirm, reject, double-verify)
  - GET  /audit   (event filtering)
"""

from __future__ import annotations

import io
import json
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import pytest_asyncio
import httpx
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport

# ---- Set TESTING flag BEFORE importing any backend modules ---------------
os.environ["TESTING"] = "1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def set_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Point LOCAL_STORE_DIR to a temp folder for the test session."""
    store_dir = tmp_path_factory.mktemp("store")
    os.environ["LOCAL_STORE_DIR"] = str(store_dir)


@pytest_asyncio.fixture(scope="module")
async def app_and_client() -> AsyncGenerator[tuple[FastAPI, AsyncClient], None]:
    """
    Create a fresh FastAPI app with an in-memory SQLite DB and yield
    (app, client).  Tables are created once per module.
    """
    from backend.main import create_app
    from backend.db.base import init_db

    test_app = create_app()

    # Initialise tables
    await init_db()

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield test_app, client


# ---------------------------------------------------------------------------
# Helper: build a minimal synthetic PNG in memory
# ---------------------------------------------------------------------------


def _make_png(width: int = 64, height: int = 64) -> bytes:
    """Return bytes of a valid tiny PNG using OpenCV."""
    import cv2

    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    @pytest.mark.anyio
    async def test_health_ok(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


class TestIngest:
    @pytest.mark.anyio
    async def test_ingest_png_success(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        png = _make_png()
        resp = await client.post(
            "/ingest",
            files={"file": ("scene_test.png", io.BytesIO(png), "image/png")},
            data={"actor": "test_user"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "image_id" in body
        assert body["filename"] == "scene_test.png"
        assert len(body["sha256"]) == 64
        assert body["width"] == 64
        assert body["height"] == 64
        assert body["is_synthetic"] is True

    @pytest.mark.anyio
    async def test_ingest_bad_mime_rejected(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.post(
            "/ingest",
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
            data={"actor": "test_user"},
        )
        assert resp.status_code == 415

    @pytest.mark.anyio
    async def test_ingest_empty_file_rejected(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.post(
            "/ingest",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
            data={"actor": "test_user"},
        )
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_ingest_creates_audit_log(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        png = _make_png()
        ingest_resp = await client.post(
            "/ingest",
            files={"file": ("audit_test.png", io.BytesIO(png), "image/png")},
            data={"actor": "audit_actor"},
        )
        assert ingest_resp.status_code == 201
        image_id = ingest_resp.json()["image_id"]

        audit_resp = await client.get(f"/audit?resource_id={image_id}&event_type=INGEST")
        assert audit_resp.status_code == 200
        entries = audit_resp.json()["entries"]
        assert len(entries) >= 1
        assert entries[0]["event_type"] == "INGEST"
        assert entries[0]["actor"] == "audit_actor"


# ---------------------------------------------------------------------------
# /infer  (inference is mocked — no .pt checkpoint needed)
# ---------------------------------------------------------------------------


MOCK_PRED = {
    "predicted_class": 0,
    "predicted_label": "building",
    "confidence": 0.87,
    "logits_json": "[1.5, 0.3, 0.2]",
    "model_arch": "SceneClassifier/MobileNetV2",
    "model_checkpoint": "ml/artifacts/best_model.pt",
    "model_sha256": "abc123" * 10 + "abcd",
}


class TestInfer:
    @pytest.mark.anyio
    async def test_infer_success(self, app_and_client: tuple) -> None:
        _, client = app_and_client

        # First ingest an image
        png = _make_png()
        ingest_resp = await client.post(
            "/ingest",
            files={"file": ("infer_me.png", io.BytesIO(png), "image/png")},
            data={"actor": "tester"},
        )
        image_id = ingest_resp.json()["image_id"]

        with patch(
            "backend.routers.infer.run_inference", return_value=MOCK_PRED
        ):
            resp = await client.post(
                "/infer",
                json={"image_id": image_id, "analyst_id": "analyst_1"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["predicted_label"] == "building"
        assert body["confidence"] == 0.87
        assert body["verified"] is False
        assert "safety_note" in body
        assert "human analyst" in body["safety_note"].lower()

    @pytest.mark.anyio
    async def test_infer_unknown_image_404(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.post(
            "/infer",
            json={"image_id": str(uuid.uuid4()), "analyst_id": "analyst_1"},
        )
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_infer_creates_audit_log(self, app_and_client: tuple) -> None:
        _, client = app_and_client

        png = _make_png()
        ingest_resp = await client.post(
            "/ingest",
            files={"file": ("infer_audit.png", io.BytesIO(png), "image/png")},
            data={"actor": "tester"},
        )
        image_id = ingest_resp.json()["image_id"]

        with patch("backend.routers.infer.run_inference", return_value=MOCK_PRED):
            infer_resp = await client.post(
                "/infer",
                json={"image_id": image_id, "analyst_id": "analyst_audit"},
            )
        inference_id = infer_resp.json()["inference_id"]

        audit_resp = await client.get(f"/audit?resource_id={inference_id}&event_type=INFER")
        entries = audit_resp.json()["entries"]
        assert len(entries) >= 1
        assert entries[0]["actor"] == "analyst_audit"


# ---------------------------------------------------------------------------
# /queue
# ---------------------------------------------------------------------------


class TestQueue:
    @pytest.mark.anyio
    async def test_queue_returns_pending(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.get("/queue?pending_only=true")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "total" in body
        assert "pending" in body
        # All returned items should be unverified
        for item in body["items"]:
            assert item["verified"] is False

    @pytest.mark.anyio
    async def test_queue_all_includes_verified(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.get("/queue?pending_only=false")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /verify
# ---------------------------------------------------------------------------


class TestVerify:
    async def _ingest_and_infer(self, client: AsyncClient) -> str:
        """Helper: ingest an image and run inference; return inference_id."""
        png = _make_png()
        ingest_resp = await client.post(
            "/ingest",
            files={"file": ("v_test.png", io.BytesIO(png), "image/png")},
            data={"actor": "tester"},
        )
        image_id = ingest_resp.json()["image_id"]

        with patch("backend.routers.infer.run_inference", return_value=MOCK_PRED):
            infer_resp = await client.post(
                "/infer",
                json={"image_id": image_id, "analyst_id": "analyst_v"},
            )
        return infer_resp.json()["inference_id"]

    @pytest.mark.anyio
    async def test_verify_confirm_success(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        inference_id = await self._ingest_and_infer(client)

        resp = await client.post(
            "/verify",
            json={
                "inference_id": inference_id,
                "analyst_id": "analyst_v",
                "confirmed": True,
                "notes": "Looks correct.",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["verified"] is True
        assert body["verified_by"] == "analyst_v"
        assert "confirmed" in body["message"]

    @pytest.mark.anyio
    async def test_verify_reject_requires_notes(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        inference_id = await self._ingest_and_infer(client)

        resp = await client.post(
            "/verify",
            json={
                "inference_id": inference_id,
                "analyst_id": "analyst_v",
                "confirmed": False,
                # notes intentionally omitted
            },
        )
        assert resp.status_code == 422  # Pydantic validation error

    @pytest.mark.anyio
    async def test_verify_reject_with_notes_success(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        inference_id = await self._ingest_and_infer(client)

        resp = await client.post(
            "/verify",
            json={
                "inference_id": inference_id,
                "analyst_id": "analyst_v",
                "confirmed": False,
                "notes": "Model misidentified open field as building.",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["verified"] is False

    @pytest.mark.anyio
    async def test_verify_double_verify_rejected(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        inference_id = await self._ingest_and_infer(client)

        # First verify
        await client.post(
            "/verify",
            json={
                "inference_id": inference_id,
                "analyst_id": "analyst_v",
                "confirmed": True,
                "notes": "ok",
            },
        )
        # Second verify — must be rejected
        resp2 = await client.post(
            "/verify",
            json={
                "inference_id": inference_id,
                "analyst_id": "analyst_v2",
                "confirmed": True,
                "notes": "trying again",
            },
        )
        assert resp2.status_code == 409

    @pytest.mark.anyio
    async def test_verify_unknown_inference_404(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.post(
            "/verify",
            json={
                "inference_id": str(uuid.uuid4()),
                "analyst_id": "analyst_v",
                "confirmed": True,
            },
        )
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_verify_creates_audit_log(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        inference_id = await self._ingest_and_infer(client)

        await client.post(
            "/verify",
            json={
                "inference_id": inference_id,
                "analyst_id": "audit_verify_actor",
                "confirmed": True,
                "notes": "verified",
            },
        )
        resp = await client.get(
            f"/audit?resource_id={inference_id}&event_type=VERIFY"
        )
        entries = resp.json()["entries"]
        assert len(entries) >= 1
        assert entries[0]["actor"] == "audit_verify_actor"


# ---------------------------------------------------------------------------
# /audit
# ---------------------------------------------------------------------------


class TestAudit:
    @pytest.mark.anyio
    async def test_audit_returns_list(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.get("/audit")
        assert resp.status_code == 200
        body = resp.json()
        assert "entries" in body
        assert "total" in body

    @pytest.mark.anyio
    async def test_audit_filter_by_event_type(self, app_and_client: tuple) -> None:
        _, client = app_and_client
        resp = await client.get("/audit?event_type=INGEST")
        assert resp.status_code == 200
        for entry in resp.json()["entries"]:
            assert entry["event_type"] == "INGEST"
