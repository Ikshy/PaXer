/**
 * src/api/client.ts
 * Typed API client wrapping all CTHMP backend endpoints.
 * Uses native fetch — no extra HTTP library needed.
 */

import type {
  IngestResponse,
  InferResponse,
  QueueResponse,
  VerifyRequest,
  VerifyResponse,
  AuditLogResponse,
} from "@/types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

// ── Helpers ───────────────────────────────────────────────────────────────

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── Ingest ────────────────────────────────────────────────────────────────

export async function ingestImage(
  file: File,
  actor = "analyst"
): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("actor", actor);
  const res = await fetch(`${BASE}/ingest`, { method: "POST", body: form });
  return handleResponse<IngestResponse>(res);
}

// ── Infer ─────────────────────────────────────────────────────────────────

export async function runInference(
  imageId: string,
  analystId = "analyst"
): Promise<InferResponse> {
  const res = await fetch(`${BASE}/infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId, analyst_id: analystId }),
  });
  return handleResponse<InferResponse>(res);
}

// ── Queue ─────────────────────────────────────────────────────────────────

export async function fetchQueue(
  pendingOnly = true,
  limit = 50,
  offset = 0
): Promise<QueueResponse> {
  const params = new URLSearchParams({
    pending_only: String(pendingOnly),
    limit: String(limit),
    offset: String(offset),
  });
  const res = await fetch(`${BASE}/queue?${params}`);
  return handleResponse<QueueResponse>(res);
}

// ── Verify ────────────────────────────────────────────────────────────────

export async function verifyInference(
  payload: VerifyRequest
): Promise<VerifyResponse> {
  const res = await fetch(`${BASE}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<VerifyResponse>(res);
}

// ── Audit ─────────────────────────────────────────────────────────────────

export async function fetchAuditLog(opts?: {
  eventType?: string;
  resourceId?: string;
  limit?: number;
  offset?: number;
}): Promise<AuditLogResponse> {
  const params = new URLSearchParams();
  if (opts?.eventType) params.set("event_type", opts.eventType);
  if (opts?.resourceId) params.set("resource_id", opts.resourceId);
  if (opts?.limit) params.set("limit", String(opts.limit));
  if (opts?.offset) params.set("offset", String(opts.offset));
  const res = await fetch(`${BASE}/audit?${params}`);
  return handleResponse<AuditLogResponse>(res);
}
