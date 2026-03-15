/**
 * src/hooks/useAuditLog.ts
 * Custom hook: fetches audit log entries with optional filters.
 */

import { useState, useEffect, useCallback } from "react";
import { fetchAuditLog } from "@/api/client";
import type { AuditLogEntry } from "@/types";

interface UseAuditLogResult {
  entries: AuditLogEntry[];
  total: number;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useAuditLog(opts?: {
  eventType?: string;
  resourceId?: string;
  limit?: number;
}): UseAuditLogResult {
  const [entries, setEntries] = useState<AuditLogEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchAuditLog(opts)
      .then((data) => {
        if (cancelled) return;
        setEntries(data.entries);
        setTotal(data.total);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(
          err instanceof Error ? err.message : "Failed to load audit log."
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opts?.eventType, opts?.resourceId, opts?.limit, tick]);

  return { entries, total, loading, error, refresh };
}
