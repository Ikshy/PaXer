/**
 * src/hooks/useQueue.ts
 * Custom hook: fetches the analyst verification queue and exposes
 * a verify action that optimistically refreshes the list.
 */

import { useState, useEffect, useCallback } from "react";
import { fetchQueue, verifyInference } from "@/api/client";
import type { QueueItem, VerifyRequest } from "@/types";

interface UseQueueResult {
  items: QueueItem[];
  pending: number;
  total: number;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  verify: (payload: VerifyRequest) => Promise<void>;
  verifying: boolean;
}

export function useQueue(pendingOnly = true): UseQueueResult {
  const [items, setItems] = useState<QueueItem[]>([]);
  const [pending, setPending] = useState(0);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [tick, setTick] = useState(0);

  const refresh = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    fetchQueue(pendingOnly)
      .then((data) => {
        if (cancelled) return;
        setItems(data.items);
        setPending(data.pending);
        setTotal(data.total);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load queue.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [pendingOnly, tick]);

  const verify = useCallback(
    async (payload: VerifyRequest) => {
      setVerifying(true);
      try {
        await verifyInference(payload);
        refresh();
      } finally {
        setVerifying(false);
      }
    },
    [refresh]
  );

  return { items, pending, total, loading, error, refresh, verify, verifying };
}
