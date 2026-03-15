/**
 * NavBar.tsx
 * Top navigation bar with tab switching and pending-count badge.
 */

import React from "react";
import { Map, ListChecks, ScrollText } from "lucide-react";
import type { Tab } from "@/types";

interface NavBarProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  pendingCount: number;
}

const TABS: { id: Tab; label: string; Icon: React.FC<{ size?: number }> }[] = [
  { id: "map", label: "Map View", Icon: Map },
  { id: "queue", label: "Analyst Queue", Icon: ListChecks },
  { id: "audit", label: "Audit Log", Icon: ScrollText },
];

const NavBar: React.FC<NavBarProps> = ({ activeTab, onTabChange, pendingCount }) => (
  <nav
    aria-label="Main navigation"
    style={{
      background: "#0f172a",
      borderBottom: "1px solid #334155",
      display: "flex",
      alignItems: "center",
      padding: "0 16px",
      height: "48px",
      gap: "4px",
    }}
  >
    <span
      style={{
        color: "#94a3b8",
        fontSize: "13px",
        fontWeight: 700,
        marginRight: "16px",
        letterSpacing: "0.05em",
        fontFamily: "monospace",
      }}
    >
      CTHMP
    </span>

    {TABS.map(({ id, label, Icon }) => {
      const isActive = activeTab === id;
      const showBadge = id === "queue" && pendingCount > 0;

      return (
        <button
          key={id}
          role="tab"
          aria-selected={isActive}
          aria-label={label}
          onClick={() => onTabChange(id)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "6px 12px",
            background: isActive ? "#1e293b" : "transparent",
            color: isActive ? "#f1f5f9" : "#64748b",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: isActive ? 600 : 400,
            position: "relative",
            transition: "background 0.15s, color 0.15s",
          }}
        >
          <Icon size={15} />
          {label}
          {showBadge && (
            <span
              aria-label={`${pendingCount} pending verifications`}
              style={{
                background: "#ef4444",
                color: "#fff",
                borderRadius: "9999px",
                fontSize: "10px",
                fontWeight: 700,
                padding: "1px 6px",
                minWidth: "18px",
                textAlign: "center",
              }}
            >
              {pendingCount}
            </span>
          )}
        </button>
      );
    })}
  </nav>
);

export default NavBar;
