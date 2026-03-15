/**
 * TimeSlider.test.tsx
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import TimeSlider from "@/components/TimeSlider";
import type { QueueItem } from "@/types";

const BASE_ITEM: QueueItem = {
  inference_id: "inf-1",
  image_id: "img-1",
  filename: "a.png",
  predicted_label: "building",
  confidence: 0.9,
  inferred_at: "2024-06-01T10:00:00Z",
  verified: false,
};

const ITEMS: QueueItem[] = [
  BASE_ITEM,
  { ...BASE_ITEM, inference_id: "inf-2", inferred_at: "2024-06-01T11:00:00Z" },
  { ...BASE_ITEM, inference_id: "inf-3", inferred_at: "2024-06-01T12:00:00Z" },
];

describe("TimeSlider", () => {
  it("renders null when items array is empty", () => {
    const { container } = render(
      <TimeSlider items={[]} cutoffMs={Date.now()} onChange={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders range input when items present", () => {
    render(
      <TimeSlider
        items={ITEMS}
        cutoffMs={new Date(ITEMS[2].inferred_at).getTime()}
        onChange={vi.fn()}
      />
    );
    expect(screen.getByRole("slider")).toBeInTheDocument();
  });

  it("calls onChange when slider moves", () => {
    const onChange = vi.fn();
    render(
      <TimeSlider
        items={ITEMS}
        cutoffMs={new Date(ITEMS[2].inferred_at).getTime()}
        onChange={onChange}
      />
    );
    const slider = screen.getByRole("slider");
    fireEvent.change(slider, {
      target: { value: String(new Date(ITEMS[1].inferred_at).getTime()) },
    });
    expect(onChange).toHaveBeenCalled();
  });

  it("shows visible item count", () => {
    const cutoff = new Date(ITEMS[1].inferred_at).getTime();
    render(
      <TimeSlider items={ITEMS} cutoffMs={cutoff} onChange={vi.fn()} />
    );
    // 2 of 3 items are at-or-before ITEMS[1]
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
  });
});
