/**
 * App.tsx
 * Root application component.
 *
 * Layout:
 *   SafetyBanner  (always visible — ethics policy)
 *   NavBar        (tab switcher with pending-count badge)
 *   <active page> (fills remaining viewport height)
 */

import React, { useState } from "react";
import SafetyBanner from "@/components/SafetyBanner";
import NavBar from "@/components/NavBar";
import MapPage from "@/pages/MapPage";
import QueuePage from "@/pages/QueuePage";
import AuditPage from "@/pages/AuditPage";
import { useQueue } from "@/hooks/useQueue";
import type { Tab } from "@/types";

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>("map");

  // Shared queue state so NavBar badge and MapPage stay in sync
  const queue = useQueue(true);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        background: "#020617",
        color: "#e2e8f0",
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        overflow: "hidden",
      }}
    >
      {/* Persistent ethics banner */}
      <SafetyBanner />

      {/* Navigation */}
      <NavBar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        pendingCount={queue.pending}
      />

      {/* Page content — fills remaining height */}
      <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
        {activeTab === "map" && (
          <MapPage items={queue.items} onRefresh={queue.refresh} />
        )}
        {activeTab === "queue" && <QueuePage />}
        {activeTab === "audit" && <AuditPage />}
      </div>

      {/* CSS keyframes for spinner animation */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        * { box-sizing: border-box; }
        body { margin: 0; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
      `}</style>
    </div>
  );
};

export default App;
