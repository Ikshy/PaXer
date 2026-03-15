/**
 * DiffPanel.tsx
 * Side-by-side "before / after" diff panel showing the selected QueueItem's
 * inference metadata vs. the raw image details.
 *
 * In a full implementation the actual image tiles would be fetched from the
 * object store.  Here we show the metadata diff with styled cards.
 */

import React from "react";
import { ImageOff, CheckCircle, XCircle, Clock } from "lucide-react";
import type { QueueItem } from "@/types";
import { LABEL_COLOURS } from "@/types";

interface DiffPanelProps {
  item: QueueItem | null;
}

const Field: React.FC<{ label: string; value: React.ReactNode }> = ({
  label,
  value,
}) => (
  <div style={{ marginBottom: "10px" }}>
    <div
      style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.08em" }}
    >
      {label}
    </div>
    <div style={{ color: "#e2e8f0", fontSize: "13px", marginTop: "2px" }}>
      {value}
    </div>
  </div>
);

const DiffPanel: React.FC<DiffPanelProps> = ({ item }) => {
  if (!item) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "#475569",
          gap: "8px",
          fontSize: "13px",
        }}
      >
        <ImageOff size={32} />
        <span>Select a marker on the map or a row in the queue to inspect</span>
      </div>
    );
  }

  const colour =
    LABEL_COLOURS[item.predicted_label as keyof typeof LABEL_COLOURS] ??
    LABEL_COLOURS.unknown;

  const StatusIcon = item.verified
    ? CheckCircle
    : Clock;
  const statusColour = item.verified ? "#22c55e" : "#f59e0b";
  const statusText = item.verified ? "Verified" : "Pending verification";

  return (
    <div style={{ padding: "16px", height: "100%", overflowY: "auto" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "16px",
          borderBottom: "1px solid #1e293b",
          paddingBottom: "12px",
        }}
      >
        <StatusIcon size={16} color={statusColour} />
        <span style={{ color: statusColour, fontSize: "12px", fontWeight: 600 }}>
          {statusText}
        </span>
      </div>

      {/* Two-column diff */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        {/* Left: Image metadata */}
        <div
          style={{
            background: "#0f172a",
            borderRadius: "8px",
            padding: "12px",
            border: "1px solid #1e293b",
          }}
        >
          <div
            style={{
              fontSize: "11px",
              color: "#475569",
              fontWeight: 700,
              marginBottom: "12px",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Source Image
          </div>

          {/* Placeholder image tile */}
          <div
            style={{
              background: "#1e293b",
              borderRadius: "6px",
              height: "120px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: "12px",
              border: "1px dashed #334155",
              color: "#475569",
              fontSize: "11px",
            }}
          >
            <div style={{ textAlign: "center" }}>
              <ImageOff size={24} style={{ marginBottom: "4px" }} />
              <div>{item.filename}</div>
            </div>
          </div>

          <Field label="Filename" value={item.filename} />
          <Field label="Image ID" value={
            <span style={{ fontFamily: "monospace", fontSize: "11px" }}>
              {item.image_id.slice(0, 18)}…
            </span>
          } />
          <Field
            label="Ingested at"
            value={new Date(item.inferred_at).toLocaleString()}
          />
        </div>

        {/* Right: Inference result */}
        <div
          style={{
            background: "#0f172a",
            borderRadius: "8px",
            padding: "12px",
            border: `1px solid ${colour}44`,
          }}
        >
          <div
            style={{
              fontSize: "11px",
              color: "#475569",
              fontWeight: 700,
              marginBottom: "12px",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Model Output
          </div>

          {/* Confidence bar */}
          <div style={{ marginBottom: "12px" }}>
            <div
              style={{
                fontSize: "10px",
                color: "#64748b",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                marginBottom: "4px",
              }}
            >
              Confidence
            </div>
            <div
              style={{
                background: "#1e293b",
                borderRadius: "9999px",
                height: "8px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${item.confidence * 100}%`,
                  height: "100%",
                  background: colour,
                  borderRadius: "9999px",
                  transition: "width 0.4s ease",
                }}
              />
            </div>
            <div
              style={{ color: colour, fontSize: "13px", fontWeight: 700, marginTop: "4px" }}
            >
              {(item.confidence * 100).toFixed(1)}%
            </div>
          </div>

          <Field
            label="Predicted label"
            value={
              <span
                style={{
                  background: colour + "22",
                  color: colour,
                  border: `1px solid ${colour}55`,
                  borderRadius: "4px",
                  padding: "2px 8px",
                  fontSize: "12px",
                  fontWeight: 600,
                }}
              >
                {item.predicted_label}
              </span>
            }
          />
          <Field
            label="Inference ID"
            value={
              <span style={{ fontFamily: "monospace", fontSize: "11px" }}>
                {item.inference_id.slice(0, 18)}…
              </span>
            }
          />
          <Field
            label="Verification status"
            value={
              <span style={{ color: statusColour, fontWeight: 600 }}>
                {statusText}
              </span>
            }
          />
        </div>
      </div>

      {/* Safety reminder */}
      <div
        role="note"
        aria-label="Safety reminder"
        style={{
          marginTop: "16px",
          background: "#1e293b",
          border: "1px solid #f59e0b44",
          borderRadius: "6px",
          padding: "10px 12px",
          color: "#fbbf24",
          fontSize: "11px",
          fontFamily: "monospace",
        }}
      >
        ⚠ This detection is a model candidate only. Human analyst sign-off via
        the Analyst Queue is required before it may be considered authoritative.
      </div>
    </div>
  );
};

export default DiffPanel;
