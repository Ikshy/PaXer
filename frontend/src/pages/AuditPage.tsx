/**
 * AuditPage.tsx
 * Audit log page — wraps AuditLogTable with the useAuditLog hook.
 */

import React, { useState } from "react";
import AuditLogTable from "@/components/AuditLogTable";
import { useAuditLog } from "@/hooks/useAuditLog";

const AuditPage: React.FC = () => {
  const [eventType, setEventType] = useState<string | undefined>(undefined);

  const { entries, total, loading, error, refresh } = useAuditLog({
    eventType,
    limit: 200,
  });

  return (
    <div style={{ height: "100%", overflow: "hidden" }}>
      <AuditLogTable
        entries={entries}
        total={total}
        loading={loading}
        error={error}
        onRefresh={refresh}
        onFilterChange={setEventType}
      />
    </div>
  );
};

export default AuditPage;
