/**
 * DiffPanel.test.tsx
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import DiffPanel from "@/components/DiffPanel";
import type { QueueItem } from "@/types";

const MOCK_ITEM: QueueItem = {
  inference_id: "inf-abc",
  image_id: "img-abc",
  filename: "scene_0005.png",
  predicted_label: "vehicle",
  confidence: 0.72,
  inferred_at: "2024-06-02T09:30:00Z",
  verified: false,
};

describe("DiffPanel", () => {
  it("shows placeholder when item is null", () => {
    render(<DiffPanel item={null} />);
    expect(screen.getByText(/select a marker/i)).toBeInTheDocument();
  });

  it("renders filename when item provided", () => {
    render(<DiffPanel item={MOCK_ITEM} />);
    expect(screen.getAllByText("scene_0005.png").length).toBeGreaterThan(0);
  });

  it("shows predicted label", () => {
    render(<DiffPanel item={MOCK_ITEM} />);
    expect(screen.getAllByText("vehicle").length).toBeGreaterThan(0);
  });

  it("shows pending status for unverified item", () => {
    render(<DiffPanel item={MOCK_ITEM} />);
    expect(screen.getAllByText(/pending verification/i).length).toBeGreaterThan(0);
  });

  it("shows verified status for verified item", () => {
    render(<DiffPanel item={{ ...MOCK_ITEM, verified: true }} />);
    expect(screen.getAllByText(/verified/i).length).toBeGreaterThan(0);
  });

  it("shows safety reminder note", () => {
    render(<DiffPanel item={MOCK_ITEM} />);
    expect(screen.getByRole("note")).toBeInTheDocument();
    expect(screen.getByText(/human analyst sign-off/i)).toBeInTheDocument();
  });

  it("displays confidence percentage", () => {
    render(<DiffPanel item={MOCK_ITEM} />);
    expect(screen.getByText(/72\.0%/)).toBeInTheDocument();
  });
});
