/**
 * VerificationPanel.test.tsx
 * Tests the ethics-gate UI: confirm/reject flows and validation.
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import VerificationPanel from "@/components/VerificationPanel";
import type { QueueItem, VerifyRequest } from "@/types";

const MOCK_ITEMS: QueueItem[] = [
  {
    inference_id: "inf-001",
    image_id: "img-001",
    filename: "scene_0001.png",
    predicted_label: "building",
    confidence: 0.87,
    inferred_at: "2024-06-01T12:00:00Z",
    verified: false,
  },
  {
    inference_id: "inf-002",
    image_id: "img-002",
    filename: "scene_0002.png",
    predicted_label: "vehicle",
    confidence: 0.65,
    inferred_at: "2024-06-01T12:05:00Z",
    verified: false,
  },
];

function renderPanel(overrides?: {
  items?: QueueItem[];
  verifying?: boolean;
  error?: string | null;
  onVerify?: (p: VerifyRequest) => Promise<void>;
}) {
  const onVerify = overrides?.onVerify ?? vi.fn().mockResolvedValue(undefined);
  const onRefresh = vi.fn();
  render(
    <VerificationPanel
      items={overrides?.items ?? MOCK_ITEMS}
      loading={false}
      verifying={overrides?.verifying ?? false}
      error={overrides?.error ?? null}
      onVerify={onVerify}
      onRefresh={onRefresh}
    />
  );
  return { onVerify, onRefresh };
}

describe("VerificationPanel", () => {
  it("renders list of pending items", () => {
    renderPanel();
    expect(screen.getByText("scene_0001.png")).toBeInTheDocument();
    expect(screen.getByText("scene_0002.png")).toBeInTheDocument();
  });

  it("shows '2 pending' count", () => {
    renderPanel();
    expect(screen.getByText(/2 pending/i)).toBeInTheDocument();
  });

  it("shows empty state when no items", () => {
    renderPanel({ items: [] });
    expect(screen.getByText(/queue is clear/i)).toBeInTheDocument();
  });

  it("shows error state", () => {
    renderPanel({ error: "Network failure" });
    expect(screen.getByText(/Network failure/i)).toBeInTheDocument();
  });

  it("shows form after selecting an item", () => {
    renderPanel();
    fireEvent.click(screen.getByText("scene_0001.png"));
    expect(screen.getByText(/analyst sign-off/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /confirm/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reject/i })).toBeInTheDocument();
  });

  it("calls onVerify with confirmed=true when Confirm clicked", async () => {
    const { onVerify } = renderPanel();
    fireEvent.click(screen.getByText("scene_0001.png"));
    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
    await waitFor(() => {
      expect(onVerify).toHaveBeenCalledWith(
        expect.objectContaining({
          inference_id: "inf-001",
          confirmed: true,
        })
      );
    });
  });

  it("blocks rejection without notes and shows error", async () => {
    const { onVerify } = renderPanel();
    fireEvent.click(screen.getByText("scene_0001.png"));
    fireEvent.click(screen.getByRole("button", { name: /reject/i }));
    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
      expect(screen.getByText(/notes are required/i)).toBeInTheDocument();
    });
    expect(onVerify).not.toHaveBeenCalled();
  });

  it("calls onVerify with confirmed=false and notes when notes provided", async () => {
    const { onVerify } = renderPanel();
    fireEvent.click(screen.getByText("scene_0001.png"));
    fireEvent.change(screen.getByLabelText(/notes/i), {
      target: { value: "False positive — open field misclassified." },
    });
    fireEvent.click(screen.getByRole("button", { name: /reject/i }));
    await waitFor(() => {
      expect(onVerify).toHaveBeenCalledWith(
        expect.objectContaining({
          inference_id: "inf-001",
          confirmed: false,
          notes: "False positive — open field misclassified.",
        })
      );
    });
  });

  it("shows immutable audit log warning", () => {
    renderPanel();
    fireEvent.click(screen.getByText("scene_0001.png"));
    expect(screen.getByText(/immutable audit log/i)).toBeInTheDocument();
  });

  it("disables buttons when verifying=true", () => {
    renderPanel({ verifying: true });
    fireEvent.click(screen.getByText("scene_0001.png"));
    expect(screen.getByRole("button", { name: /confirm/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /reject/i })).toBeDisabled();
  });
});
