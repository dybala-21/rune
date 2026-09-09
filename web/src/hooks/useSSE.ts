import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SSE_EVENT_TYPES, type SseEventType } from '../types';
import { ensureWebAuth, resetWebAuth, setClientId as setApiClientId } from '../api';
import { fetchRunSnapshot, getLiveSessionId } from '../api';
import { RunRecovery } from '../utils/runRecovery';

export interface SseConnection {
  connected: boolean;
  clientId: string | null;
  addEventListener: (event: SseEventType, handler: (data: unknown) => void) => () => void;
  refresh: () => void;
}

export function useSSE(): SseConnection {
  const [connected, setConnected] = useState(false);
  const [clientId, setClientId] = useState<string | null>(null);
  const listenersRef = useRef(new Map<SseEventType, Set<(data: unknown) => void>>());
  const sourceRef = useRef<EventSource | null>(null);
  const restoreRef = useRef<() => void>(() => {});
  const refresh = useCallback(() => restoreRef.current(), []);

  useEffect(() => {
    let disposed = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let recoveryVersion = 0;
    const emit = (type: SseEventType, data: unknown) => {
      for (const handler of listenersRef.current.get(type) ?? []) handler(data);
    };
    const recovery = new RunRecovery(emit);
    const restore = async () => {
      const version = ++recoveryVersion;
      const sessionId = getLiveSessionId();
      recovery.begin();
      try {
        const { run, available } = await fetchRunSnapshot(sessionId);
        if (disposed || version !== recoveryVersion) return;
        if (sessionId !== getLiveSessionId() || !recovery.finish(available === false ? undefined : run)) {
          void restore();
        }
      } catch {
        if (disposed || version !== recoveryVersion) return;
        sourceRef.current?.close();
        setConnected(false);
        scheduleReconnect();
      }
    };

    restoreRef.current = () => { void restore(); };

    // EventSource stops retrying once CLOSED; open a new stream after a backoff.
    const scheduleReconnect = () => {
      if (disposed || retryTimer) return;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        void connect();
      }, 2000);
    };

    const connect = async () => {
      if (disposed) return;
      try {
        await ensureWebAuth();
      } catch {
        if (!disposed) {
          setConnected(false);
          scheduleReconnect();
        }
        return;
      }
      if (disposed) return;

      const source = new EventSource('/api/events', { withCredentials: true });
      sourceRef.current = source;

      source.addEventListener('connected', (e) => {
        try {
          const data = JSON.parse((e as MessageEvent).data) as { clientId: string };
          setClientId(data.clientId);
          setApiClientId(data.clientId);
          setConnected(true);
          void restore();
        } catch { /* ignore */ }
      });

      for (const eventType of SSE_EVENT_TYPES.filter(e => e !== 'connected')) {
        source.addEventListener(eventType, (e) => {
          try {
            const data = JSON.parse((e as MessageEvent).data);
            if (eventType === 'resync_required') void restore();
            else recovery.receive(eventType, data);
          } catch { /* ignore parse errors */ }
        });
      }

      source.onerror = () => {
        setConnected(false);
        // Leave CONNECTING alone while the browser retries.
        if (source.readyState === EventSource.CLOSED) {
          source.close();
          if (sourceRef.current === source) sourceRef.current = null;
          // Refresh auth in case the session expired.
          resetWebAuth();
          scheduleReconnect();
        }
      };

      source.onopen = () => {
        // Wait for `connected` to supply the client ID.
      };
    };

    void connect();

    return () => {
      disposed = true;
      restoreRef.current = () => {};
      recoveryVersion += 1;
      if (retryTimer) clearTimeout(retryTimer);
      sourceRef.current?.close();
      sourceRef.current = null;
      setConnected(false);
      setClientId(null);
    };
  }, []);

  const addEventListener = useCallback((event: SseEventType, handler: (data: unknown) => void) => {
    if (!listenersRef.current.has(event)) {
      listenersRef.current.set(event, new Set());
    }
    listenersRef.current.get(event)!.add(handler);

    return () => {
      listenersRef.current.get(event)?.delete(handler);
    };
  }, []);

  return useMemo(
    () => ({ connected, clientId, addEventListener, refresh }),
    [connected, clientId, addEventListener, refresh],
  );
}
