/**
 * NavBar.test.tsx
 */

import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import NavBar from "@/components/NavBar";

describe("NavBar", () => {
  const noop = vi.fn();

  it("renders all three tabs", () => {
    render(<NavBar activeTab="map" onTabChange={noop} pendingCount={0} />);
    expect(screen.getByRole("tab", { name: /map view/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /analyst queue/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /audit log/i })).toBeInTheDocument();
  });

  it("calls onTabChange with correct tab id", () => {
    const onChange = vi.fn();
    render(<NavBar activeTab="map" onTabChange={onChange} pendingCount={0} />);
    fireEvent.click(screen.getByRole("tab", { name: /analyst queue/i }));
    expect(onChange).toHaveBeenCalledWith("queue");
  });

  it("shows pending badge when pendingCount > 0", () => {
    render(<NavBar activeTab="map" onTabChange={noop} pendingCount={7} />);
    expect(screen.getByLabelText(/7 pending verifications/i)).toBeInTheDocument();
  });

  it("hides badge when pendingCount is 0", () => {
    render(<NavBar activeTab="map" onTabChange={noop} pendingCount={0} />);
    expect(screen.queryByLabelText(/pending verifications/i)).toBeNull();
  });

  it("marks active tab as aria-selected=true", () => {
    render(<NavBar activeTab="audit" onTabChange={noop} pendingCount={0} />);
    const auditTab = screen.getByRole("tab", { name: /audit log/i });
    expect(auditTab).toHaveAttribute("aria-selected", "true");
  });

  it("marks inactive tabs as aria-selected=false", () => {
    render(<NavBar activeTab="audit" onTabChange={noop} pendingCount={0} />);
    const mapTab = screen.getByRole("tab", { name: /map view/i });
    expect(mapTab).toHaveAttribute("aria-selected", "false");
  });
});
