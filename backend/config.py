"""
backend/config.py
=================
Centralised application settings loaded from environment variables.
No credentials are hardcoded — see .env.example for required variables.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


class Settings:
    """Application settings resolved from environment variables."""

    # --- App ---
    app_env: str = os.getenv("APP_ENV", "development")
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-change-me")

    # --- Database ---
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "cthmp")
    postgres_user: str = os.getenv("POSTGRES_USER", "cthmp_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "changeme")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Synchronous URL used in Alembic migrations and tests."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # --- Object store ---
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY", "changeme")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "cthmp-imagery")

    # --- ML model ---
    model_checkpoint: str = os.getenv("MODEL_CHECKPOINT", "ml/artifacts/best_model.pt")
    model_arch: str = os.getenv("MODEL_ARCH", "SceneClassifier/MobileNetV2")

    # --- Local imagery store (used when MinIO is not configured) ---
    local_store_dir: Path = Path(os.getenv("LOCAL_STORE_DIR", "data/store"))

    # --- Synthetic data ---
    synth_output_dir: str = os.getenv("SYNTH_OUTPUT_DIR", "data")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
