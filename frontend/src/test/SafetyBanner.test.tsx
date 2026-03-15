/**
 * SafetyBanner.test.tsx
 * Tests that the ethics banner renders and contains required text.
 * This is a hard requirement — the banner must always be visible.
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import SafetyBanner from "@/components/SafetyBanner";

describe("SafetyBanner", () => {
  it("renders without crashing", () => {
    render(<SafetyBanner />);
  });

  it("displays RESEARCH USE ONLY notice", () => {
    render(<SafetyBanner />);
    expect(screen.getByText(/RESEARCH USE ONLY/i)).toBeInTheDocument();
  });

  it("mentions humanitarian use", () => {
    render(<SafetyBanner />);
    expect(screen.getByText(/humanitarian/i)).toBeInTheDocument();
  });

  it("mentions human analyst sign-off requirement", () => {
    render(<SafetyBanner />);
    expect(screen.getByText(/human analyst sign-off/i)).toBeInTheDocument();
  });

  it("has correct ARIA banner role", () => {
    render(<SafetyBanner />);
    expect(screen.getByRole("banner")).toBeInTheDocument();
  });
});
