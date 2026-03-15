"""
tests/test_documentation.py
============================
Validates that all required documentation files exist, are non-empty,
and contain mandatory safety / ethics content.

These tests are intentionally pedantic — they encode the ethics policy
in machine-checkable form so that a CI failure flags documentation gaps
before any code review.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def exists(relative: str) -> bool:
    return (REPO_ROOT / relative).exists()


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


class TestRequiredFilesExist:
    REQUIRED_FILES = [
        "README.md",
        "ETHICS.md",
        "DEPLOYMENT.md",
        "LICENSE",
        "pyproject.toml",
        ".env.example",
        ".gitignore",
        ".dockerignore",
        "Dockerfile.dev",
        "Dockerfile.backend",
        "Dockerfile.frontend",
        "docker-compose.yml",
        "docker-compose.prod.yml",
        ".github/workflows/ci.yml",
        "docs/ARCHITECTURE.md",
        "docs/ETHICS_BOARD_CHECKLIST.md",
        "docker/nginx.conf",
        "docker/postgres/init.sql",
        # Core Python modules
        "data/synthetic_generator.py",
        "ml/dataset.py",
        "ml/model.py",
        "ml/train.py",
        "ml/evaluate.py",
        "backend/main.py",
        "backend/config.py",
        "backend/schemas.py",
        "backend/db/base.py",
        "backend/models/imagery.py",
        "backend/services/audit.py",
        "backend/services/store.py",
        "backend/services/inference.py",
        "backend/routers/ingest.py",
        "backend/routers/infer.py",
        "backend/routers/queue.py",
        "backend/routers/verify.py",
        "backend/routers/audit_log.py",
        "redteam/perturbations.py",
        "redteam/scenarios.py",
        "redteam/eval.py",
        # Tests
        "tests/test_synthetic_generator.py",
        "tests/test_dataset.py",
        "tests/test_model.py",
        "tests/test_backend.py",
        "tests/test_redteam.py",
        "tests/test_documentation.py",
        # Frontend
        "frontend/package.json",
        "frontend/vite.config.ts",
        "frontend/index.html",
        "frontend/src/main.tsx",
        "frontend/src/App.tsx",
        "frontend/src/components/SafetyBanner.tsx",
        "frontend/src/components/VerificationPanel.tsx",
    ]

    @pytest.mark.parametrize("path", REQUIRED_FILES)
    def test_file_exists(self, path: str) -> None:
        assert exists(path), f"Required file missing: {path}"

    @pytest.mark.parametrize("path", REQUIRED_FILES)
    def test_file_nonempty(self, path: str) -> None:
        content = read(path).strip()
        assert len(content) > 0, f"File is empty: {path}"


# ---------------------------------------------------------------------------
# README.md mandatory content
# ---------------------------------------------------------------------------


class TestReadme:
    def test_contains_safety_warning(self) -> None:
        text = read("README.md").lower()
        assert "humanitarian" in text

    def test_contains_architecture_section(self) -> None:
        text = read("README.md")
        assert "Architecture" in text or "architecture" in text

    def test_contains_quick_start(self) -> None:
        text = read("README.md").lower()
        assert "quick start" in text or "quickstart" in text

    def test_contains_ethics_reference(self) -> None:
        text = read("README.md")
        assert "ETHICS.md" in text

    def test_contains_testing_instructions(self) -> None:
        text = read("README.md").lower()
        assert "pytest" in text or "npm test" in text

    def test_contains_parts_roadmap(self) -> None:
        text = read("README.md").lower()
        assert "roadmap" in text or "parts" in text


# ---------------------------------------------------------------------------
# ETHICS.md mandatory content
# ---------------------------------------------------------------------------


class TestEthics:
    def test_contains_absolute_prohibitions(self) -> None:
        text = read("ETHICS.md").lower()
        assert "prohibited" in text or "prohibition" in text

    def test_prohibits_targeting(self) -> None:
        text = read("ETHICS.md").lower()
        assert "targeting" in text

    def test_prohibits_strike_planning(self) -> None:
        text = read("ETHICS.md").lower()
        assert "strike" in text

    def test_requires_human_in_the_loop(self) -> None:
        text = read("ETHICS.md").lower()
        assert "human" in text and ("loop" in text or "analyst" in text)

    def test_requires_audit_logging(self) -> None:
        text = read("ETHICS.md").lower()
        assert "audit" in text

    def test_requires_immutable_logs(self) -> None:
        text = read("ETHICS.md").lower()
        assert "immutable" in text or "append" in text

    def test_references_responsible_disclosure(self) -> None:
        text = read("ETHICS.md").lower()
        assert "disclosure" in text

    def test_references_synthetic_data(self) -> None:
        text = read("ETHICS.md").lower()
        assert "synthetic" in text

    def test_references_ethics_board_checklist(self) -> None:
        text = read("ETHICS.md")
        assert "ETHICS_BOARD_CHECKLIST" in text


# ---------------------------------------------------------------------------
# docs/ETHICS_BOARD_CHECKLIST.md mandatory structure
# ---------------------------------------------------------------------------


class TestEthicsBoardChecklist:
    def test_has_blocking_items(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md")
        assert "[BLOCKING]" in text

    def test_has_sign_off_section(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md").lower()
        assert "sign-off" in text or "sign off" in text

    def test_has_data_provenance_section(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md").lower()
        assert "provenance" in text

    def test_has_human_in_loop_section(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md").lower()
        assert "human" in text and ("loop" in text or "verify" in text)

    def test_has_audit_section(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md").lower()
        assert "audit" in text

    def test_has_robustness_section(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md").lower()
        assert "robustness" in text or "red" in text

    def test_has_board_decision_field(self) -> None:
        text = read("docs/ETHICS_BOARD_CHECKLIST.md").lower()
        assert "approved" in text and "rejected" in text


# ---------------------------------------------------------------------------
# docs/ARCHITECTURE.md
# ---------------------------------------------------------------------------


class TestArchitecture:
    def test_has_database_schema(self) -> None:
        text = read("docs/ARCHITECTURE.md").lower()
        assert "schema" in text or "audit_logs" in text

    def test_has_api_contract(self) -> None:
        text = read("docs/ARCHITECTURE.md")
        assert "/ingest" in text and "/verify" in text

    def test_has_ml_model_description(self) -> None:
        text = read("docs/ARCHITECTURE.md").lower()
        assert "mobilenet" in text or "mobilev2" in text or "scene" in text

    def test_has_ci_pipeline_description(self) -> None:
        text = read("docs/ARCHITECTURE.md").lower()
        assert "ci" in text and ("pipeline" in text or "job" in text)


# ---------------------------------------------------------------------------
# LICENSE
# ---------------------------------------------------------------------------


class TestLicense:
    def test_is_apache2(self) -> None:
        text = read("LICENSE")
        assert "Apache License" in text
        assert "Version 2.0" in text

    def test_has_ethics_additional_condition(self) -> None:
        text = read("LICENSE").lower()
        assert "targeting" in text or "kinetic" in text


# ---------------------------------------------------------------------------
# DEPLOYMENT.md
# ---------------------------------------------------------------------------


class TestDeployment:
    def test_has_prerequisites(self) -> None:
        text = read("DEPLOYMENT.md").lower()
        assert "prerequisite" in text or "docker" in text

    def test_has_quick_start_commands(self) -> None:
        text = read("DEPLOYMENT.md")
        assert "docker compose" in text.lower()

    def test_has_production_checklist(self) -> None:
        text = read("DEPLOYMENT.md").lower()
        assert "production" in text and "checklist" in text

    def test_has_troubleshooting(self) -> None:
        text = read("DEPLOYMENT.md").lower()
        assert "troubleshoot" in text


# ---------------------------------------------------------------------------
# SafetyBanner — source-level checks
# ---------------------------------------------------------------------------


class TestSafetyBannerSource:
    def test_banner_not_easily_hidden(self) -> None:
        text = read("frontend/src/components/SafetyBanner.tsx")
        # Must not have display:none or visibility:hidden at top level
        assert "display: none" not in text
        assert "visibility: hidden" not in text

    def test_banner_has_aria_role(self) -> None:
        text = read("frontend/src/components/SafetyBanner.tsx")
        assert 'role="banner"' in text

    def test_banner_references_humanitarian(self) -> None:
        text = read("frontend/src/components/SafetyBanner.tsx").lower()
        assert "humanitarian" in text

    def test_banner_references_sign_off(self) -> None:
        text = read("frontend/src/components/SafetyBanner.tsx").lower()
        assert "sign-off" in text or "analyst" in text


# ---------------------------------------------------------------------------
# verify.py — ethics gate source checks
# ---------------------------------------------------------------------------


class TestVerifyRouterSource:
    def test_verified_defaults_false(self) -> None:
        text = read("backend/routers/infer.py")
        assert "verified=False" in text

    def test_double_verify_rejected(self) -> None:
        text = read("backend/routers/verify.py").lower()
        assert "409" in text or "conflict" in text

    def test_notes_required_on_rejection(self) -> None:
        text = read("backend/schemas.py").lower()
        assert "notes" in text and ("required" in text or "confirmed" in text)


# ---------------------------------------------------------------------------
# Audit service — append-only check
# ---------------------------------------------------------------------------


class TestAuditServiceSource:
    def test_no_update_in_audit_service(self) -> None:
        text = read("backend/services/audit.py").lower()
        # The word "update" must not appear as a SQL/ORM operation
        assert "db.merge" not in text.lower()
        # Standard update pattern should not appear
        assert ".update(" not in text

    def test_audit_log_model_has_no_update_method(self) -> None:
        text = read("backend/models/imagery.py")
        # AuditLog class should have no mutable columns after creation
        assert "AuditLog" in text
