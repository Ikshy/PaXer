# Conflict Transparency & Humanitarian Monitoring Platform (CTHMP)

> **⚠ Mandatory safety notice** — See [ETHICS.md](ETHICS.md) before using this software.  
> This platform is for **humanitarian monitoring and transparency research only**.  
> Operational targeting, strike planning, or military advantage use is **strictly prohibited**.

---

## What is CTHMP?

CTHMP is a research-grade, open-source pipeline for:

- Generating **synthetic satellite-style imagery** for ML research (no real imagery required)
- Training a **lightweight scene classifier** (MobileNetV2 backbone) on that synthetic data
- Exposing a **FastAPI backend** that ingests images, runs inference, queues results for human review, and keeps an immutable audit trail
- Providing a **React frontend** with a Leaflet map view, time slider, side-by-side diff panel, and analyst verification UI
- Running **robustness red-team evaluations** (9 perturbation types × 3 severity levels × 12 scenarios)

Every detection is a **candidate only**. No result is authoritative until a human analyst signs off via the `/verify` endpoint.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser / Analyst                            │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP
          ┌──────────────▼──────────────┐
          │       Frontend              │
          │  React 18 + Vite            │
          │  ┌──────────┐ ┌──────────┐  │
          │  │ Map View │ │ Queue    │  │
          │  │ Leaflet  │ │ Verify   │  │
          │  │ TimeSlide│ │ Audit Log│  │
          │  └──────────┘ └──────────┘  │
          │  SafetyBanner (always on)   │
          └──────────────┬──────────────┘
                         │ fetch /ingest /infer /queue /verify /audit
                         │
          ┌──────────────▼──────────────┐
          │       Backend               │
          │  FastAPI + Uvicorn          │
          │                             │
          │  POST /ingest  ──────────────────► Local FS / MinIO
          │  POST /infer   ──┐                (image bytes)
          │  GET  /queue     │
          │  POST /verify  ◄─┘ SceneClassifier
          │  GET  /audit       MobileNetV2    ◄── ml/artifacts/best_model.pt
          │  GET  /health      (PyTorch)
          └──────┬──────────────────────┘
                 │ SQLAlchemy async
          ┌──────▼──────────────────────┐
          │       PostgreSQL            │
          │  image_records              │
          │  inference_results          │  ← verified=False until /verify
          │  audit_logs (immutable)     │
          └─────────────────────────────┘

          ┌─────────────────────────────┐
          │  Offline Research Tools     │
          │  data/synthetic_generator   │  Deterministic synthetic imagery
          │  ml/train.py                │  MobileNetV2 training loop
          │  ml/evaluate.py             │  Per-class P/R/F1 report
          │  redteam/eval.py            │  Robustness harness (12 scenarios)
          └─────────────────────────────┘
```

### Data flow

```
synthetic_generator.py
        │
        ▼ PNG + COCO JSON
  data/samples/
        │
        ▼
   ml/train.py  ──────────────────────► ml/artifacts/best_model.pt
        │                                        │
        ▼                                        ▼
  ml/evaluate.py                    backend/services/inference.py
  (offline eval)                    (live inference via /infer)
                                             │
                                             ▼
                                    InferenceResult (verified=False)
                                             │
                                             ▼ human analyst
                                    POST /verify → verified=True
                                             │
                                             ▼
                                    AuditLog entry (immutable)
```

---

## Repository layout

```
cthmp/
├── data/
│   ├── synthetic_generator.py   Deterministic synthetic imagery + COCO annotations
│   └── samples/                 Pre-generated sample output (10 scenes)
├── ml/
│   ├── dataset.py               PyTorch Dataset (COCO loader, majority-label)
│   ├── model.py                 SceneClassifier (MobileNetV2 + custom head)
│   ├── train.py                 Training loop, checkpointing, provenance JSON
│   └── evaluate.py              Standalone evaluation script
├── backend/
│   ├── main.py                  FastAPI app factory
│   ├── config.py                Settings from env vars
│   ├── schemas.py               Pydantic v2 request/response models
│   ├── db/base.py               Async SQLAlchemy engine
│   ├── models/imagery.py        ORM: ImageRecord, InferenceResult, AuditLog
│   ├── services/
│   │   ├── audit.py             Immutable audit log writer
│   │   ├── store.py             Local FS object store (MinIO-swappable)
│   │   └── inference.py        Cached model loader + forward pass
│   └── routers/
│       ├── ingest.py            POST /ingest
│       ├── infer.py             POST /infer
│       ├── queue.py             GET  /queue
│       ├── verify.py            POST /verify  ← ethics gate
│       └── audit_log.py         GET  /audit
├── frontend/
│   └── src/
│       ├── components/          SafetyBanner, NavBar, MapView, DiffPanel,
│       │                        TimeSlider, VerificationPanel, AuditLogTable
│       ├── pages/               MapPage, QueuePage, AuditPage
│       ├── hooks/               useQueue, useAuditLog
│       ├── api/client.ts        Typed fetch wrappers
│       └── types/index.ts       Shared TypeScript types
├── redteam/
│   ├── perturbations.py         9 perturbation functions × 3 severities
│   ├── scenarios.py             12 structured test scenarios
│   └── eval.py                  Robustness harness + JSON report writer
├── tests/                       pytest: 100+ tests across all modules
├── docker/
│   ├── nginx.conf               SPA routing + API proxy
│   └── postgres/init.sql        DB initialisation
├── Dockerfile.dev               Dev container (source-mounted)
├── Dockerfile.backend           Production backend (multi-stage)
├── Dockerfile.frontend          Production frontend (Nginx)
├── docker-compose.yml           Full local stack
├── docker-compose.prod.yml      Production overrides
├── .github/workflows/ci.yml     3-job CI (Python · Frontend · Docker)
├── pyproject.toml               Python project config + deps
├── ETHICS.md                    Ethics & usage policy
├── DEPLOYMENT.md                Local and production deployment guide
└── docs/
    ├── ARCHITECTURE.md          Detailed architecture reference
    └── ETHICS_BOARD_CHECKLIST.md  Review board checklist
```

---

## Quick start

```bash
# 1. Clone
git clone <repo-url> && cd cthmp

# 2. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Generate synthetic data
python data/synthetic_generator.py

# 4. Train model (~2 min on CPU)
python ml/train.py

# 5. Full stack via Docker Compose
cp .env.example .env
docker compose up --build

# 6. Open browser
#   Frontend:  http://localhost:5173
#   API docs:  http://localhost:8000/docs
#   MinIO:     http://localhost:9001
```

---

## Running tests

```bash
# All Python tests (100+ assertions)
TESTING=1 pytest tests/ -v

# Frontend tests (36 Vitest tests)
cd frontend && npm test

# Red-team evaluation (12 scenarios)
python redteam/eval.py
```

---

## Parts roadmap

| Part | Status | Description |
|------|--------|-------------|
| 1 | ✅ | Repo skeleton, synthetic data generator, CI |
| 2 | ✅ | ML training (MobileNetV2), evaluation script |
| 3 | ✅ | FastAPI backend, DB models, audit logging |
| 4 | ✅ | React frontend: map, diff, verification panel |
| 5 | ✅ | Dockerfiles, docker-compose, CI pipeline |
| 6 | ✅ | Robustness red-team harness (9 perturbations, 12 scenarios) |
| 7 | ✅ | Full docs, ethics policy, architecture diagram |

---

## Contributing

1. Fork the repository and create a feature branch.
2. Ensure all tests pass: `TESTING=1 pytest tests/ -v && cd frontend && npm test`.
3. Run the red-team harness and include results in your PR description.
4. Confirm your changes comply with `ETHICS.md` — offensive-use scenarios will be rejected.
5. Open a pull request against `develop`.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use CTHMP in research, please cite:

```bibtex
@software{cthmp2024,
  title  = {Conflict Transparency \& Humanitarian Monitoring Platform},
  year   = {2024},
  note   = {Research-grade platform for humanitarian monitoring. Apache 2.0.}
}
```
