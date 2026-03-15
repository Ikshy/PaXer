/**
 * AuditLogTable.tsx
 * Read-only table view of the immutable audit log.
 * Supports filtering by event type.
 */

import React, { useState } from "react";
import { RefreshCw, AlertTriangle } from "lucide-react";
import type { AuditLogEntry } from "@/types";

interface AuditLogTableProps {
  entries: AuditLogEntry[];
  total: number;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  onFilterChange: (eventType: string | undefined) => void;
}

const EVENT_TYPES = ["", "INGEST", "INFER", "VERIFY", "SYSTEM"];

const EVENT_COLOURS: Record<string, string> = {
  INGEST: "#6366f1",
  INFER: "#06b6d4",
  VERIFY: "#22c55e",
  SYSTEM: "#94a3b8",
};

const AuditLogTable: React.FC<AuditLogTableProps> = ({
  entries,
  total,
  loading,
  error,
  onRefresh,
  onFilterChange,
}) => {
  const [filter, setFilter] = useState("");

  const handleFilter = (v: string) => {
    setFilter(v);
    onFilterChange(v || undefined);
  };

  return (
    <div
      style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}
    >
      {/* Toolbar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "12px",
          padding: "10px 16px",
          borderBottom: "1px solid #1e293b",
          flexShrink: 0,
        }}
      >
        <span style={{ color: "#64748b", fontSize: "12px" }}>
          {total} total entries
        </span>
        <select
          aria-label="Filter by event type"
          value={filter}
          onChange={(e) => handleFilter(e.target.value)}
          style={{
            background: "#0f172a",
            border: "1px solid #334155",
            color: "#e2e8f0",
            borderRadius: "4px",
            padding: "4px 8px",
            fontSize: "12px",
          }}
        >
          {EVENT_TYPES.map((t) => (
            <option key={t} value={t}>
              {t || "All events"}
            </option>
          ))}
        </select>
        <button
          onClick={onRefresh}
          aria-label="Refresh audit log"
          style={{
            background: "transparent",
            border: "none",
            color: "#64748b",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: "4px",
            fontSize: "12px",
          }}
        >
          <RefreshCw size={13} />
          Refresh
        </button>
      </div>

      {/* Table area */}
      {error ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "20px 16px",
            color: "#ef4444",
            fontSize: "13px",
          }}
        >
          <AlertTriangle size={16} />
          {error}
        </div>
      ) : loading ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "20px 16px",
            color: "#64748b",
            fontSize: "13px",
          }}
        >
          <RefreshCw size={16} color="#6366f1" />
          Loading…
        </div>
      ) : (
        <div style={{ overflowY: "auto", flex: 1 }}>
          <table
            aria-label="Audit log"
            style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}
          >
            <thead>
              <tr style={{ background: "#0f172a", position: "sticky", top: 0 }}>
                {["#", "Event", "Actor", "Resource", "Timestamp"].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: "left",
                      padding: "8px 12px",
                      color: "#475569",
                      fontWeight: 600,
                      borderBottom: "1px solid #1e293b",
                      letterSpacing: "0.05em",
                      textTransform: "uppercase",
                      fontSize: "10px",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entries.length === 0 ? (
                <tr>
                  <td
                    colSpan={5}
                    style={{ padding: "20px 12px", color: "#475569", textAlign: "center" }}
                  >
                    No entries found.
                  </td>
                </tr>
              ) : (
                entries.map((entry) => {
                  const colour = EVENT_COLOURS[entry.event_type] ?? "#94a3b8";
                  return (
                    <tr
                      key={entry.id}
                      style={{ borderBottom: "1px solid #0f172a" }}
                    >
                      <td style={tdStyle}>{entry.id}</td>
                      <td style={tdStyle}>
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
                          {entry.event_type}
                        </span>
                      </td>
                      <td style={tdStyle}>{entry.actor}</td>
                      <td
                        style={{
                          ...tdStyle,
                          fontFamily: "monospace",
                          fontSize: "10px",
                          color: "#64748b",
                        }}
                      >
                        {entry.resource_type} / {entry.resource_id.slice(0, 12)}…
                      </td>
                      <td style={{ ...tdStyle, color: "#64748b", whiteSpace: "nowrap" }}>
                        {new Date(entry.timestamp_utc).toLocaleString()}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

const tdStyle: React.CSSProperties = {
  padding: "8px 12px",
  color: "#e2e8f0",
  verticalAlign: "middle",
};

export default AuditLogTable;
