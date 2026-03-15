/**
 * MapPage.tsx
 * Map view: Leaflet map + time slider + diff panel side-by-side.
 */

import React, { useState, useMemo } from "react";
import MapView from "@/components/MapView";
import TimeSlider from "@/components/TimeSlider";
import DiffPanel from "@/components/DiffPanel";
import UploadButton from "@/components/UploadButton";
import type { QueueItem } from "@/types";

interface MapPageProps {
  items: QueueItem[];
  onRefresh: () => void;
}

const MapPage: React.FC<MapPageProps> = ({ items, onRefresh }) => {
  const [selectedItem, setSelectedItem] = useState<QueueItem | null>(null);
  const maxMs = useMemo(() => {
    if (items.length === 0) return Date.now();
    return Math.max(...items.map((i) => new Date(i.inferred_at).getTime()));
  }, [items]);

  const [cutoffMs, setCutoffMs] = useState<number>(maxMs);

  // Keep cutoffMs in sync when items change
  React.useEffect(() => setCutoffMs(maxMs), [maxMs]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 16px",
          borderBottom: "1px solid #1e293b",
          flexShrink: 0,
          background: "#0f172a",
        }}
      >
        <span style={{ color: "#64748b", fontSize: "12px" }}>
          {items.length} inference result{items.length !== 1 ? "s" : ""} loaded
        </span>
        <UploadButton onComplete={onRefresh} />
      </div>

      {/* Time slider */}
      <TimeSlider items={items} cutoffMs={cutoffMs} onChange={setCutoffMs} />

      {/* Map + diff panel */}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Map — 65% width */}
        <div style={{ flex: "0 0 65%", position: "relative" }}>
          <MapView
            items={items}
            cutoffMs={cutoffMs}
            onSelectItem={setSelectedItem}
          />
        </div>

        {/* Diff panel — 35% width */}
        <div
          style={{
            flex: "0 0 35%",
            borderLeft: "1px solid #1e293b",
            background: "#0f172a",
            overflowY: "auto",
          }}
        >
          <DiffPanel item={selectedItem} />
        </div>
      </div>
    </div>
  );
};

export default MapPage;
