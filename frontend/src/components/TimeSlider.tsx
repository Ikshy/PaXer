/**
 * TimeSlider.tsx
 * Horizontal range slider to filter map markers by inferred_at timestamp.
 * Drives parent state; pure controlled component.
 */

import React, { useMemo } from "react";
import type { QueueItem } from "@/types";

interface TimeSliderProps {
  items: QueueItem[];
  /** Unix-ms timestamp: only items at-or-before this are shown. */
  cutoffMs: number;
  onChange: (cutoffMs: number) => void;
}

const fmt = (iso: string) =>
  new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

const TimeSlider: React.FC<TimeSliderProps> = ({ items, cutoffMs, onChange }) => {
  const { minMs, maxMs } = useMemo(() => {
    if (items.length === 0) return { minMs: 0, maxMs: Date.now() };
    const ts = items.map((i) => new Date(i.inferred_at).getTime());
    return { minMs: Math.min(...ts), maxMs: Math.max(...ts) };
  }, [items]);

  if (items.length === 0) return null;

  const visibleCount = items.filter(
    (i) => new Date(i.inferred_at).getTime() <= cutoffMs
  ).length;

  return (
    <div
      aria-label="Time filter slider"
      style={{
        background: "#0f172a",
        borderBottom: "1px solid #334155",
        padding: "10px 16px",
        display: "flex",
        alignItems: "center",
        gap: "12px",
      }}
    >
      <span style={{ color: "#64748b", fontSize: "11px", whiteSpace: "nowrap" }}>
        {fmt(items.reduce((a, b) =>
          new Date(a.inferred_at) < new Date(b.inferred_at) ? a : b
        ).inferred_at)}
      </span>

      <input
        type="range"
        aria-label="Time cutoff"
        min={minMs}
        max={maxMs}
        step={Math.max(1, Math.floor((maxMs - minMs) / 100))}
        value={cutoffMs}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ flex: 1, accentColor: "#6366f1" }}
      />

      <span style={{ color: "#64748b", fontSize: "11px", whiteSpace: "nowrap" }}>
        {fmt(items.reduce((a, b) =>
          new Date(a.inferred_at) > new Date(b.inferred_at) ? a : b
        ).inferred_at)}
      </span>

      <span
        aria-live="polite"
        style={{
          color: "#94a3b8",
          fontSize: "11px",
          whiteSpace: "nowrap",
          minWidth: "60px",
          textAlign: "right",
        }}
      >
        {visibleCount} / {items.length}
      </span>
    </div>
  );
};

export default TimeSlider;
