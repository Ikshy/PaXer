# CTHMP Architecture Reference

## Overview

CTHMP is a seven-layer research platform. Each layer has a single
responsibility and communicates with adjacent layers through typed interfaces.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 7 — Documentation & Ethics                                        │
│  README.md · ETHICS.md · DEPLOYMENT.md · docs/                          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Layer 6 — Red-Team Harness                                              │
│  redteam/perturbations.py · redteam/scenarios.py · redteam/eval.py      │
│  9 perturbations · 3 severity levels · 12 scenarios                     │
│  Outputs: redteam/reports/robustness_report_*.json                       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Layer 5 — Containerisation & CI                                         │
│  Dockerfile.dev · Dockerfile.backend · Dockerfile.frontend               │
│  docker-compose.yml · docker-compose.prod.yml                            │
│  .github/workflows/ci.yml  (3 jobs: Python · Frontend · Docker)         │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Layer 4 — Frontend                                                      │
│  React 18 + Vite + TypeScript + Leaflet                                  │
│  Components: SafetyBanner · NavBar · MapView · TimeSlider                │
│              DiffPanel · VerificationPanel · AuditLogTable               │
│  Hooks: useQueue · useAuditLog                                           │
│  API client: src/api/client.ts (typed fetch wrappers)                   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │ HTTP (fetch / Vite proxy)
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Layer 3 — Backend                                                       │
│  FastAPI + Uvicorn + SQLAlchemy (async) + Pydantic v2                   │
│                                                                          │
│  Endpoints:                                                              │
│    POST /ingest  → validate → store → ImageRecord → AuditLog            │
│    POST /infer   → load → inference → InferenceResult → AuditLog        │
│    GET  /queue   → pending InferenceResults for analyst                  │
│    POST /verify  → analyst sign-off → verified=True → AuditLog          │
│    GET  /audit   → read-only AuditLog view                              │
│    GET  /health  → liveness probe                                        │
│                                                                          │
│  Services:                                                               │
│    audit.py    — append-only event logging (never update/delete)        │
│    store.py    — local FS / MinIO abstraction                           │
│    inference.py — cached model loader + forward pass                    │
└────────────┬─────────────────────────────┬──────────────────────────────┘
             │ asyncpg                      │ local FS / MinIO S3
┌────────────▼────────────┐   ┌────────────▼────────────────────────────┐
│  PostgreSQL             │   │  Object Store                            │
│  image_records          │   │  data/store/<uuid>/<filename>.png       │
│  inference_results      │   │  (MinIO in production)                  │
│  audit_logs (append)    │   └─────────────────────────────────────────┘
└─────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Layer 2 — ML                                                            │
│  ml/dataset.py   — SyntheticSceneDataset (COCO → majority label)        │
│  ml/model.py     — SceneClassifier (MobileNetV2 backbone, 3 classes)    │
│  ml/train.py     — Training loop, checkpointing, provenance JSON        │
│  ml/evaluate.py  — Per-class P/R/F1 standalone report                  │
│                                                                          │
│  Classes: building (0) · vehicle (1) · open_area (2)                   │
│  Output:  ml/artifacts/best_model.pt + model_provenance.json           │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Layer 1 — Synthetic Data                                                │
│  data/synthetic_generator.py                                             │
│  · Deterministic (fixed seed)                                            │
│  · Generates (H,W,3) uint8 PNG images + COCO JSON annotations          │
│  · 3 object types: rectangle (building) · ellipse (vehicle)             │
│                      polygon (open_area)                                 │
│  Output: data/samples/images/*.png + data/samples/annotations.json     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Database schema

```sql
-- image_records
id           VARCHAR(36) PK
filename     VARCHAR(255)
store_path   VARCHAR(512)
sha256       VARCHAR(64)   INDEXED
width        INTEGER
height       INTEGER
is_synthetic BOOLEAN       DEFAULT TRUE
source_note  TEXT
ingested_at  TIMESTAMPTZ
ingested_by  VARCHAR(128)

-- inference_results
id               VARCHAR(36) PK
image_id         VARCHAR(36) FK → image_records(id)  INDEXED
predicted_class  INTEGER
predicted_label  VARCHAR(64)
confidence       FLOAT
logits_json      TEXT          (JSON array of raw logits)
model_arch       VARCHAR(128)
model_checkpoint VARCHAR(512)
model_sha256     VARCHAR(64)
verified         BOOLEAN       DEFAULT FALSE  ← ethics gate
verified_by      VARCHAR(128)  NULLABLE
verified_at      TIMESTAMPTZ   NULLABLE
analyst_notes    TEXT          NULLABLE
inferred_at      TIMESTAMPTZ

-- audit_logs  (APPEND ONLY — never UPDATE or DELETE)
id             SERIAL PK
event_type     VARCHAR(32)   INDEXED  (INGEST|INFER|VERIFY|SYSTEM)
actor          VARCHAR(128)
resource_type  VARCHAR(64)
resource_id    VARCHAR(36)   INDEXED
detail         TEXT          (JSON)
timestamp_utc  TIMESTAMPTZ   INDEXED
```

---

## API contract

### POST /ingest
```
Request:  multipart/form-data
  file  : PNG/JPEG/TIFF image (≤ 20 MB)
  actor : string (operator id for audit log)

Response 201:
  image_id    : UUID
  filename    : string
  sha256      : hex64
  width/height: int
  ingested_at : ISO-8601
```

### POST /infer
```
Request:  application/json
  image_id   : UUID (from /ingest)
  analyst_id : string

Response 200:
  inference_id    : UUID
  predicted_label : "building"|"vehicle"|"open_area"
  confidence      : float 0–1
  logits          : [float, float, float]
  model_sha256    : hex64
  verified        : false          ← always false on creation
  safety_note     : string         ← always present
```

### POST /verify
```
Request:  application/json
  inference_id : UUID
  analyst_id   : string
  confirmed    : bool
  notes        : string (required when confirmed=false)

Response 200:
  verified    : bool
  verified_by : string
  verified_at : ISO-8601
  message     : string

Errors:
  409 — already verified (immutable)
  422 — rejected without notes
  404 — inference not found
```

---

## ML model

| Property | Value |
|---|---|
| Architecture | MobileNetV2 + custom head |
| Input | 256×256 RGB float32 (ImageNet normalisation) |
| Output | 3-class logits → softmax probabilities |
| Parameters | ~2.3M |
| Classes | building (0), vehicle (1), open_area (2) |
| Training data | Synthetic only (seeded generator) |
| Checkpoint format | PyTorch state-dict (.pt) |
| Provenance | SHA-256 of checkpoint + hyperparams in model_provenance.json |

---

## Red-team scenarios

| Scenario | Perturbation(s) | Simulates |
|---|---|---|
| clean | none | Baseline |
| mild_noise | gaussian_noise σ=5 | Low sensor noise |
| severe_noise | gaussian_noise σ=60 | Heavy degradation |
| transmission_artefacts | jpeg q=10 | Bandwidth-limited downlink |
| sensor_blur | gaussian_blur k=7 | Atmospheric/defocus |
| cloud_occlusion_mild | occlusion 10% | Partial cloud cover |
| cloud_occlusion_severe | occlusion 40% | Heavy cloud |
| brightness_overexposed | brightness +100 | Overexposure |
| low_contrast | contrast ×0.2 | Haze/fog |
| sensor_tilt | rotation 45° | Off-nadir angle |
| combined_degraded | blur+noise+brightness | Multiple failures |
| worst_case | noise+occlusion+jpeg | Extreme degradation |

---

## CI pipeline

```
push/PR to main or develop
        │
        ├── Job 1: Python
        │     ruff lint → mypy typecheck → pytest (≥60% coverage)
        │
        ├── Job 2: Frontend
        │     tsc typecheck → vitest (36 tests + coverage)
        │
        └── Job 3: Docker
              build Dockerfile.backend → start container
              → poll /health → assert 200 OK
              build Dockerfile.frontend
```
