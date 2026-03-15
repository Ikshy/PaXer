/**
 * QueuePage.tsx
 * Analyst queue page — wraps VerificationPanel with the useQueue hook.
 */

import React from "react";
import VerificationPanel from "@/components/VerificationPanel";
import { useQueue } from "@/hooks/useQueue";

const QueuePage: React.FC = () => {
  const { items, loading, verifying, error, verify, refresh } = useQueue(true);

  return (
    <div style={{ height: "100%", overflow: "hidden" }}>
      <VerificationPanel
        items={items}
        loading={loading}
        verifying={verifying}
        error={error}
        onVerify={verify}
        onRefresh={refresh}
      />
    </div>
  );
};

export default QueuePage;
