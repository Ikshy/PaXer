/**
 * MapView.tsx
 * Leaflet map displaying synthetic scene inference results as circle markers.
 *
 * NOTE: Coordinates are fully synthetic/randomised around a neutral research
 * bounding box — no real location data is used or implied.
 * Markers use a seeded pseudo-random offset so positions are deterministic
 * per inference_id.
 */

import React, { useMemo } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Tooltip,
  ZoomControl,
} from "react-leaflet";
import type { QueueItem } from "@/types";
import { LABEL_COLOURS } from "@/types";

// Neutral research centre (no operational significance)
const MAP_CENTRE: [number, number] = [48.8566, 2.3522]; // Paris — arbitrary demo
const MAP_ZOOM = 11;

/**
 * Deterministic pseudo-random offset from inference_id.
 * Keeps the same marker in the same place across refreshes.
 */
function pseudoOffset(id: string, scale: number): number {
  let hash = 0;
  for (let i = 0; i < id.length; i++) {
    hash = (Math.imul(31, hash) + id.charCodeAt(i)) | 0;
  }
  return ((hash & 0xffff) / 0xffff - 0.5) * scale;
}

interface MapViewProps {
  items: QueueItem[];
  /** Unix-ms cutoff from time slider */
  cutoffMs: number;
  onSelectItem: (item: QueueItem) => void;
}

const MapView: React.FC<MapViewProps> = ({ items, cutoffMs, onSelectItem }) => {
  const visible = useMemo(
    () => items.filter((i) => new Date(i.inferred_at).getTime() <= cutoffMs),
    [items, cutoffMs]
  );

  return (
    <MapContainer
      center={MAP_CENTRE}
      zoom={MAP_ZOOM}
      zoomControl={false}
      style={{ height: "100%", width: "100%", background: "#1e293b" }}
      aria-label="Synthetic scene inference map"
    >
      <ZoomControl position="topright" />
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {visible.map((item) => {
        const lat =
          MAP_CENTRE[0] + pseudoOffset(item.inference_id + "lat", 0.18);
        const lng =
          MAP_CENTRE[1] + pseudoOffset(item.inference_id + "lng", 0.25);
        const colour =
          LABEL_COLOURS[item.predicted_label as keyof typeof LABEL_COLOURS] ??
          LABEL_COLOURS.unknown;
        const radius = 8 + item.confidence * 8;

        return (
          <CircleMarker
            key={item.inference_id}
            center={[lat, lng]}
            radius={radius}
            pathOptions={{
              color: item.verified ? "#22c55e" : colour,
              fillColor: colour,
              fillOpacity: 0.7,
              weight: item.verified ? 2 : 1,
              dashArray: item.verified ? undefined : "4 2",
            }}
            eventHandlers={{ click: () => onSelectItem(item) }}
          >
            <Tooltip>
              <div style={{ fontSize: "12px" }}>
                <strong>{item.predicted_label}</strong>
                <br />
                Confidence: {(item.confidence * 100).toFixed(1)}%
                <br />
                {item.verified ? "✅ Verified" : "⏳ Pending verification"}
                <br />
                <span style={{ color: "#9ca3af", fontSize: "10px" }}>
                  {item.filename}
                </span>
              </div>
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
};

export default MapView;
