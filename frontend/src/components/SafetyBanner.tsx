/**
 * SafetyBanner.tsx
 * Persistent top banner reminding users of the platform's ethical constraints.
 * Must remain visible at all times per the ethics policy.
 */

import React from "react";
import { ShieldAlert } from "lucide-react";

const SafetyBanner: React.FC = () => (
  <div
    role="banner"
    aria-label="Safety notice"
    style={{
      background: "#1e293b",
      borderBottom: "2px solid #f59e0b",
      color: "#fef3c7",
      padding: "6px 16px",
      fontSize: "12px",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      fontFamily: "monospace",
      letterSpacing: "0.02em",
    }}
  >
    <ShieldAlert size={14} color="#f59e0b" aria-hidden />
    <span>
      <strong>CTHMP — RESEARCH USE ONLY.</strong> Humanitarian &amp;
      transparency purposes only. All detections require human analyst
      sign-off. No operational targeting use permitted.
    </span>
  </div>
);

export default SafetyBanner;
