/**
 * apiClient.test.ts
 * Tests the API client error-handling logic using fetch mocks.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// Mock fetch globally before importing client
const mockFetch = vi.fn();
global.fetch = mockFetch;

// Re-import after mock is in place
import { fetchQueue, verifyInference } from "@/api/client";

function mockOk(body: unknown) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: async () => body,
  } as Response);
}

function mockError(status: number, text: string) {
  mockFetch.mockResolvedValueOnce({
    ok: false,
    status,
    statusText: text,
    text: async () => text,
  } as Response);
}

beforeEach(() => mockFetch.mockReset());
afterEach(() => vi.restoreAllMocks());

describe("fetchQueue", () => {
  it("returns queue data on success", async () => {
    const payload = { items: [], total: 0, pending: 0 };
    mockOk(payload);
    const result = await fetchQueue();
    expect(result).toEqual(payload);
  });

  it("includes pending_only param in URL", async () => {
    mockOk({ items: [], total: 0, pending: 0 });
    await fetchQueue(true);
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain("pending_only=true");
  });

  it("throws on HTTP error", async () => {
    mockError(500, "Internal Server Error");
    await expect(fetchQueue()).rejects.toThrow("HTTP 500");
  });
});

describe("verifyInference", () => {
  it("sends correct JSON payload", async () => {
    mockOk({
      inference_id: "x",
      verified: true,
      verified_by: "a",
      verified_at: "2024-01-01T00:00:00Z",
      analyst_notes: null,
      message: "ok",
    });

    await verifyInference({
      inference_id: "inf-123",
      analyst_id: "analyst_a",
      confirmed: true,
    });

    const [, options] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(options.body as string);
    expect(body.inference_id).toBe("inf-123");
    expect(body.confirmed).toBe(true);
  });

  it("throws on HTTP 409 conflict", async () => {
    mockError(409, "Already verified");
    await expect(
      verifyInference({
        inference_id: "inf-dup",
        analyst_id: "a",
        confirmed: true,
      })
    ).rejects.toThrow("HTTP 409");
  });

  it("throws on HTTP 404", async () => {
    mockError(404, "Not found");
    await expect(
      verifyInference({
        inference_id: "bad-id",
        analyst_id: "a",
        confirmed: true,
      })
    ).rejects.toThrow("HTTP 404");
  });
});
