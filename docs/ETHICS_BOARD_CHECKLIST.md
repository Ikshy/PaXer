# Ethics Review Board Checklist — CTHMP

**Platform**: Conflict Transparency & Humanitarian Monitoring Platform  
**Version under review**: ___________  
**Review date**: ___________  
**Reviewers**: ___________  
**Institution**: ___________

---

## Instructions

This checklist must be completed in full before any deployment of CTHMP
against real (non-synthetic) imagery, or before any publication of results
derived from the platform.

Mark each item: ✅ Pass | ❌ Fail | ⚠ Partial | N/A Not applicable

A single ❌ on items marked **[BLOCKING]** must halt deployment until resolved.

---

## Section A — Use case review

| # | Item | Status | Notes |
|---|------|--------|-------|
| A1 | **[BLOCKING]** The stated use case is explicitly humanitarian, transparency, or academic research — NOT military targeting, strike planning, or offensive operations | | |
| A2 | **[BLOCKING]** The deployment organisation has no active role in armed conflict that could benefit from this system's outputs | | |
| A3 | The use case has been reviewed against ETHICS.md Section 2 (Absolute prohibitions) and no prohibited use applies | | |
| A4 | A named responsible officer has accepted accountability for ethical use | | |
| A5 | There is a documented mechanism for ceasing operations if prohibited use is detected | | |

---

## Section B — Data provenance

| # | Item | Status | Notes |
|---|------|--------|-------|
| B1 | **[BLOCKING]** All training data is either fully synthetic (CTHMP generator) or from clearly public/open sources with cited URLs | | |
| B2 | No classified, restricted, or commercially licensed imagery is used | | |
| B3 | `ImageRecord.source_note` is populated with provenance information for every non-synthetic image | | |
| B4 | `ImageRecord.is_synthetic = True` is set correctly for all synthetic images | | |
| B5 | The SHA-256 of the annotations file is recorded in `model_provenance.json` | | |
| B6 | No personally identifiable information (faces, plates, addresses) is present in any ingested imagery | | |

---

## Section C — Human-in-the-loop verification

| # | Item | Status | Notes |
|---|------|--------|-------|
| C1 | **[BLOCKING]** `InferenceResult.verified` defaults to `False` in the codebase — confirm this has not been changed | | |
| C2 | **[BLOCKING]** No downstream system or dashboard treats `verified=False` results as authoritative | | |
| C3 | Analysts have received documented training on the verification workflow | | |
| C4 | A minimum of one independent analyst must verify each result before it is used in any report or decision | | |
| C5 | The analyst sign-off UI (VerificationPanel) displays the safety warning ("immutable audit log") and requires confirmation | | |
| C6 | Rejection of a result requires analyst notes explaining the reason — confirmed this is enforced at both API (422) and UI level | | |
| C7 | Average analyst review time per detection is documented and deemed adequate for meaningful human oversight | | |

---

## Section D — Audit trail integrity

| # | Item | Status | Notes |
|---|------|--------|-------|
| D1 | **[BLOCKING]** No UPDATE or DELETE operations exist on the `audit_logs` table — confirmed by code review of `backend/services/audit.py` | | |
| D2 | Every INGEST event is logged before the function returns — confirmed by reviewing `backend/routers/ingest.py` | | |
| D3 | Every INFER event is logged before the function returns | | |
| D4 | Every VERIFY event is logged before the function returns | | |
| D5 | The audit log is accessible to the review board on request | | |
| D6 | Audit log backup procedures are documented and tested | | |
| D7 | Audit log timestamps are in UTC and verified against an NTP source | | |

---

## Section E — Model provenance

| # | Item | Status | Notes |
|---|------|--------|-------|
| E1 | `ml/artifacts/model_provenance.json` exists and contains: timestamp, seed, hyperparams, data SHA-256, model arch | | |
| E2 | `InferenceResult.model_sha256` is populated for every inference result | | |
| E3 | The checkpoint file hash can be independently verified: `sha256sum ml/artifacts/best_model.pt` | | |
| E4 | The model was trained exclusively on synthetic or approved public data (cross-reference with Section B) | | |
| E5 | `safety_note` field is present in every `InferResponse` and displayed in the UI | | |

---

## Section F — Robustness evaluation

| # | Item | Status | Notes |
|---|------|--------|-------|
| F1 | The red-team harness (`redteam/eval.py`) has been run against the deployed model | | |
| F2 | `redteam/reports/robustness_report_latest.json` is available for review | | |
| F3 | The report documents accuracy degradation under `worst_case` scenario — reviewed and understood | | |
| F4 | Acceptable FPR (false alarm rate) thresholds have been defined per class and compared against report results | | |
| F5 | Acceptable FNR (missed detection rate) thresholds have been defined per class and compared against results | | |
| F6 | Operating procedures describe how analysts should treat low-confidence detections (confidence < defined threshold) | | |

---

## Section G — Technical safeguards

| # | Item | Status | Notes |
|---|------|--------|-------|
| G1 | `SafetyBanner` component is visible in all frontend views — confirmed via browser testing | | |
| G2 | `SafetyBanner` tests pass (`npm test` → SafetyBanner.test.tsx) | | |
| G3 | All Python tests pass: `TESTING=1 pytest tests/ -v` | | |
| G4 | All frontend tests pass: `cd frontend && npm test` | | |
| G5 | No hardcoded credentials in the codebase — confirmed by running `grep -r "password\|secret\|key" --include="*.py" --include="*.ts" | grep -v ".env\|example\|test\|comment"` | | |
| G6 | Production deployment uses HTTPS — TLS certificate verified | | |
| G7 | Database ports (5432) and object store ports (9000) are not exposed externally | | |
| G8 | Access to the `/verify` endpoint is restricted to authenticated analysts | | |

---

## Section H — Incident response

| # | Item | Status | Notes |
|---|------|--------|-------|
| H1 | A documented incident response plan exists for suspected misuse | | |
| H2 | The plan includes a procedure to immediately suspend inference capabilities | | |
| H3 | Contact details for the responsible disclosure channel are documented | | |
| H4 | The team is aware of the 90-day coordinated disclosure policy for security vulnerabilities | | |

---

## Section I — Ongoing governance

| # | Item | Status | Notes |
|---|------|--------|-------|
| I1 | A review cycle has been scheduled (recommended: every 6 months or on major model update) | | |
| I2 | Any change to the model checkpoint triggers a new provenance record and this checklist review | | |
| I3 | Any change to Section 2 or Section 3 of ETHICS.md triggers a full board review | | |
| I4 | Results of this review are archived and available for audit | | |

---

## Sign-off

By signing below, each reviewer confirms they have independently assessed
the items in their assigned sections and that all BLOCKING items are marked ✅.

| Reviewer | Role | Sections reviewed | Signature | Date |
|----------|------|-------------------|-----------|------|
| | | | | |
| | | | | |
| | | | | |

**Board decision**:

- [ ] **Approved** — deployment may proceed
- [ ] **Conditional approval** — deployment may proceed subject to: ___________
- [ ] **Deferred** — the following issues must be resolved first: ___________
- [ ] **Rejected** — reasons: ___________

**Next scheduled review date**: ___________

---

*This checklist was developed in accordance with CTHMP ETHICS.md v1.0.*
