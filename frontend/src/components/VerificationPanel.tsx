/**
 * VerificationPanel.tsx
 * Analyst verification panel — the ethics gate in the UI.
 * Analyst must confirm or reject each inference result with optional notes.
 * Rejection without notes is blocked (matching backend validation).
 */

import React, { useState } from "react";
import { CheckCircle, XCircle, RefreshCw, AlertTriangle } from "lucide-react";
import type { QueueItem, VerifyRequest } from "@/types";
import { LABEL_COLOURS } from "@/types";

interface VerificationPanelProps {
  items: QueueItem[];
  loading: boolean;
  verifying: boolean;
  error: string | null;
  onVerify: (payload: VerifyRequest) => Promise<void>;
  onRefresh: () => void;
}

const ANALYST_ID = "analyst_ui"; // In production, from auth context

const VerificationPanel: React.FC<VerificationPanelProps> = ({
  items,
  loading,
  verifying,
  error,
  onVerify,
  onRefresh,
}) => {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [formError, setFormError] = useState<string | null>(null);

  const handleVerify = async (confirmed: boolean) => {
    if (!selectedId) return;
    if (!confirmed && !notes.trim()) {
      setFormError("Notes are required when rejecting a result.");
      return;
    }
    setFormError(null);
    await onVerify({
      inference_id: selectedId,
      analyst_id: ANALYST_ID,
      confirmed,
      notes: notes.trim() || undefined,
    });
    setSelectedId(null);
    setNotes("");
  };

  if (loading) {
    return (
      <div style={centreStyle}>
        <RefreshCw size={20} color="#6366f1" style={{ animation: "spin 1s linear infinite" }} />
        <span style={{ color: "#64748b", fontSize: "13px" }}>Loading queue…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div style={centreStyle}>
        <AlertTriangle size={20} color="#ef4444" />
        <span style={{ color: "#ef4444", fontSize: "13px" }}>{error}</span>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div style={centreStyle}>
        <CheckCircle size={20} color="#22c55e" />
        <span style={{ color: "#64748b", fontSize: "13px" }}>
          No pending verifications. Queue is clear.
        </span>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>
      {/* Item list */}
      <div
        style={{
          width: "260px",
          borderRight: "1px solid #1e293b",
          overflowY: "auto",
          flexShrink: 0,
        }}
      >
        <div
          style={{
            padding: "10px 12px",
            borderBottom: "1px solid #1e293b",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span style={{ color: "#94a3b8", fontSize: "12px", fontWeight: 600 }}>
            {items.length} pending
          </span>
          <button
            onClick={onRefresh}
            aria-label="Refresh queue"
            style={iconBtnStyle}
          >
            <RefreshCw size={13} />
          </button>
        </div>

        {items.map((item) => {
          const colour =
            LABEL_COLOURS[item.predicted_label as keyof typeof LABEL_COLOURS] ??
            LABEL_COLOURS.unknown;
          const isSelected = item.inference_id === selectedId;

          return (
            <button
              key={item.inference_id}
              onClick={() => {
                setSelectedId(item.inference_id);
                setNotes("");
                setFormError(null);
              }}
              aria-selected={isSelected}
              style={{
                width: "100%",
                textAlign: "left",
                padding: "10px 12px",
                background: isSelected ? "#1e293b" : "transparent",
                border: "none",
                borderLeft: `3px solid ${isSelected ? colour : "transparent"}`,
                cursor: "pointer",
                borderBottom: "1px solid #0f172a",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "3px" }}>
                <span
                  style={{
                    background: colour + "22",
                    color: colour,
                    border: `1px solid ${colour}55`,
                    borderRadius: "3px",
                    padding: "1px 6px",
                    fontSize: "10px",
                    fontWeight: 700,
                  }}
                >
                  {item.predicted_label}
                </span>
                <span style={{ color: colour, fontSize: "11px" }}>
                  {(item.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ color: "#64748b", fontSize: "10px", fontFamily: "monospace" }}>
                {item.filename}
              </div>
              <div style={{ color: "#334155", fontSize: "10px", marginTop: "2px" }}>
                {new Date(item.inferred_at).toLocaleString()}
              </div>
            </button>
          );
        })}
      </div>

      {/* Verification form */}
      <div style={{ flex: 1, padding: "20px", overflowY: "auto" }}>
        {!selectedId ? (
          <div style={centreStyle}>
            <span style={{ color: "#475569", fontSize: "13px" }}>
              Select an item from the list to verify
            </span>
          </div>
        ) : (
          <>
            <h2
              style={{
                color: "#e2e8f0",
                fontSize: "14px",
                fontWeight: 600,
                marginBottom: "16px",
              }}
            >
              Analyst Sign-off
            </h2>

            <div
              role="note"
              style={{
                background: "#1e293b",
                border: "1px solid #f59e0b44",
                borderRadius: "6px",
                padding: "10px 12px",
                color: "#fbbf24",
                fontSize: "11px",
                fontFamily: "monospace",
                marginBottom: "16px",
              }}
            >
              ⚠ Your decision will be recorded in the immutable audit log.
              Only confirm if you have independently reviewed this detection.
            </div>

            <label
              htmlFor="analyst-notes"
              style={{
                display: "block",
                color: "#94a3b8",
                fontSize: "12px",
                marginBottom: "6px",
              }}
            >
              Notes{" "}
              <span style={{ color: "#475569" }}>(required if rejecting)</span>
            </label>
            <textarea
              id="analyst-notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={4}
              placeholder="Add your assessment notes here…"
              style={{
                width: "100%",
                background: "#0f172a",
                border: "1px solid #334155",
                borderRadius: "6px",
                color: "#e2e8f0",
                padding: "10px",
                fontSize: "13px",
                resize: "vertical",
                fontFamily: "inherit",
                boxSizing: "border-box",
              }}
            />

            {formError && (
              <div
                role="alert"
                style={{
                  color: "#ef4444",
                  fontSize: "12px",
                  marginTop: "6px",
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                }}
              >
                <AlertTriangle size={12} />
                {formError}
              </div>
            )}

            <div style={{ display: "flex", gap: "10px", marginTop: "16px" }}>
              <button
                onClick={() => handleVerify(true)}
                disabled={verifying}
                aria-label="Confirm detection"
                style={{
                  ...actionBtnStyle,
                  background: "#16a34a",
                  opacity: verifying ? 0.5 : 1,
                }}
              >
                <CheckCircle size={14} />
                Confirm
              </button>
              <button
                onClick={() => handleVerify(false)}
                disabled={verifying}
                aria-label="Reject detection"
                style={{
                  ...actionBtnStyle,
                  background: "#dc2626",
                  opacity: verifying ? 0.5 : 1,
                }}
              >
                <XCircle size={14} />
                Reject
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// ── Shared micro-styles ────────────────────────────────────────────────────

const centreStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  height: "100%",
  gap: "10px",
};

const iconBtnStyle: React.CSSProperties = {
  background: "transparent",
  border: "none",
  color: "#64748b",
  cursor: "pointer",
  padding: "4px",
  borderRadius: "4px",
  display: "flex",
  alignItems: "center",
};

const actionBtnStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "6px",
  padding: "8px 16px",
  border: "none",
  borderRadius: "6px",
  color: "#fff",
  fontWeight: 600,
  fontSize: "13px",
  cursor: "pointer",
};

export default VerificationPanel;
