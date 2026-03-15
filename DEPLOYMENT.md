# CTHMP Deployment Guide

> **Safety reminder**: this platform is for humanitarian monitoring and
> transparency research only. Review `ETHICS.md` before any deployment.

---

## Prerequisites

| Tool | Minimum version |
|------|----------------|
| Docker | 24.x |
| Docker Compose | v2.x (`docker compose`, not `docker-compose`) |
| Python | 3.11+ (for running outside Docker) |
| Node.js | 20 LTS (for running frontend outside Docker) |
| Git | any recent |

---

## Quick-start: Full local stack via Docker Compose

```bash
# 1. Clone
git clone <repo-url> && cd cthmp

# 2. Copy and edit environment file
cp .env.example .env
# Edit .env — change POSTGRES_PASSWORD, SECRET_KEY at minimum.

# 3. Generate synthetic training data
docker compose --profile tools run synth

# 4. Train the model (writes ml/artifacts/best_model.pt)
#    Run this on the host (needs GPU or ~5 min on CPU):
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python data/synthetic_generator.py
python ml/train.py

# 5. Start the full stack
docker compose up --build

# Services now available:
#   http://localhost:5173   — Frontend (Vite dev server)
#   http://localhost:8000/docs — Backend API (Swagger UI)
#   http://localhost:9001   — MinIO console  (admin/changeme)
```

---

## Step-by-step service startup order

Docker Compose handles this automatically via `depends_on` + healthchecks,
but here is the manual order if you need to start services individually:

```bash
# 1. Database first
docker compose up -d postgres
docker compose exec postgres pg_isready   # wait for healthy

# 2. Object store
docker compose up -d minio

# 3. Backend (waits for postgres healthcheck)
docker compose up -d backend

# 4. Frontend
docker compose up -d frontend
```

---

## Running tests

### Python tests (all parts)
```bash
# Outside Docker (recommended for fast iteration):
TESTING=1 pytest tests/ -v --tb=short

# Inside Docker:
docker compose --profile tools run test
```

### Frontend tests
```bash
cd frontend
npm install
npm test                  # single run
npm run test:watch        # watch mode
npm run test:coverage     # with coverage report
```

---

## Environment variables reference

All variables are documented in `.env.example`. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | **Change this** | `changeme` |
| `SECRET_KEY` | **Change this** | `dev-secret-change-me` |
| `MINIO_SECRET_KEY` | **Change this** | `changeme` |
| `TESTING` | Set to `1` to use SQLite in-memory DB | `0` |
| `LOCAL_STORE_DIR` | Where images are stored (dev) | `data/store` |
| `MODEL_CHECKPOINT` | Path to trained `.pt` file | `ml/artifacts/best_model.pt` |

---

## Production deployment

### 1. Build production images

```bash
# From repo root:
docker compose -f docker-compose.yml -f docker-compose.prod.yml build
```

### 2. Set secure environment variables

```bash
cp .env.example .env.prod
# Edit .env.prod:
#   - Set strong POSTGRES_PASSWORD, MINIO_SECRET_KEY, SECRET_KEY
#   - Set APP_ENV=production
#   - Set MODEL_CHECKPOINT to your trained checkpoint path
```

### 3. Deploy

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.prod.yml \
  --env-file .env.prod \
  up -d
```

### Production checklist

- [ ] All secrets in `.env.prod` are long random strings (not `changeme`)
- [ ] `APP_ENV=production` is set (disables wildcard CORS)
- [ ] Postgres data volume is backed up regularly
- [ ] MinIO bucket versioning is enabled
- [ ] `ml/artifacts/best_model.pt` exists and is mounted
- [ ] HTTPS termination is handled by a reverse proxy (e.g., Caddy, Traefik)
- [ ] Firewall: only ports 80/443 exposed externally; 5432/9000/8000 are internal-only
- [ ] Ethics review board sign-off obtained (see `ETHICS.md`)

---

## Regenerating synthetic data and retraining

```bash
# Generate more synthetic scenes (e.g. 100 images):
python data/synthetic_generator.py --num-images 100 --seed 42

# Retrain:
python ml/train.py --epochs 20 --batch-size 8

# Evaluate:
python ml/evaluate.py --checkpoint ml/artifacts/best_model.pt
```

---

## Architecture diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Host / Browser                        │
└───────────┬─────────────────────────────────────────────────┘
            │ HTTP :5173 (dev) / :8080 (prod)
            ▼
┌─────────────────────┐
│  Frontend           │  React + Vite (dev) / Nginx (prod)
│  - Map view         │
│  - Time slider      │
│  - Diff panel       │
│  - Verification UI  │
│  - Audit log        │
└────────┬────────────┘
         │ fetch /ingest /infer /queue /verify /audit
         │ HTTP :8000
         ▼
┌─────────────────────┐     ┌──────────────────┐
│  Backend (FastAPI)  │────▶│  PostgreSQL :5432 │
│  - /ingest          │     │  image_records    │
│  - /infer           │     │  inference_results│
│  - /queue           │     │  audit_logs       │
│  - /verify          │     └──────────────────┘
│  - /audit           │
│  - /health          │     ┌──────────────────┐
│                     │────▶│  Local FS / MinIO │
│  SceneClassifier    │     │  (image store)    │
│  (MobileNetV2)      │     └──────────────────┘
└─────────────────────┘

┌─────────────────────┐
│  data/ (synthetic)  │  Offline — no real imagery
│  synthetic_generator│
│  ml/train.py        │
└─────────────────────┘
```

---

## Troubleshooting

**Backend fails to start: "Model checkpoint not found"**
```bash
python ml/train.py   # generates ml/artifacts/best_model.pt
```

**Postgres connection refused**
```bash
docker compose logs postgres   # check for startup errors
docker compose ps              # verify postgres is healthy
```

**Frontend blank page / API 502**
```bash
# Check backend is running:
curl http://localhost:8000/health
# Check Vite proxy config in vite.config.ts matches your backend port
```

**Tests fail with "annotations not found"**
```bash
python data/synthetic_generator.py   # regenerate data/samples/
```
