/**
 * src/types/index.ts
 * Shared TypeScript interfaces matching the FastAPI Pydantic schemas.
 */

// ── Ingest ────────────────────────────────────────────────────────────────

export interface IngestResponse {
  image_id: string;
  filename: string;
  sha256: string;
  width: number;
  height: number;
  is_synthetic: boolean;
  store_path: string;
  ingested_at: string; // ISO-8601
  message: string;
}

// ── Inference ─────────────────────────────────────────────────────────────

export interface InferResponse {
  inference_id: string;
  image_id: string;
  predicted_class: number;
  predicted_label: string;
  confidence: number;
  logits: number[];
  model_arch: string;
  model_checkpoint: string;
  model_sha256: string;
  verified: boolean;
  inferred_at: string;
  safety_note: string;
}

// ── Queue ─────────────────────────────────────────────────────────────────

export interface QueueItem {
  inference_id: string;
  image_id: string;
  filename: string;
  predicted_label: string;
  confidence: number;
  inferred_at: string;
  verified: boolean;
}

export interface QueueResponse {
  items: QueueItem[];
  total: number;
  pending: number;
}

// ── Verify ────────────────────────────────────────────────────────────────

export interface VerifyRequest {
  inference_id: string;
  analyst_id: string;
  confirmed: boolean;
  notes?: string;
}

export interface VerifyResponse {
  inference_id: string;
  verified: boolean;
  verified_by: string;
  verified_at: string;
  analyst_notes: string | null;
  message: string;
}

// ── Audit ─────────────────────────────────────────────────────────────────

export interface AuditLogEntry {
  id: number;
  event_type: string;
  actor: string;
  resource_type: string;
  resource_id: string;
  detail: string;
  timestamp_utc: string;
}

export interface AuditLogResponse {
  entries: AuditLogEntry[];
  total: number;
}

// ── UI state ──────────────────────────────────────────────────────────────

export type Tab = "map" | "queue" | "audit";

export type LabelColour = "building" | "vehicle" | "open_area" | "unknown";

export const LABEL_COLOURS: Record<LabelColour, string> = {
  building: "#6366f1",
  vehicle: "#22c55e",
  open_area: "#f59e0b",
  unknown: "#94a3b8",
};
