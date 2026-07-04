import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SseEventType } from '../types';
import { ensureWebAuth, resetWebAuth, setClientId as setApiClientId } from '../api';

export interface SseConnection {
  connected: boolean;
  clientId: string | null;
  addEventListener: (event: SseEventType, handler: (data: unknown) => void) => () => void;
}

export function useSSE(): SseConnection {
  const [connected, setConnected] = useState(false);
  const [clientId, setClientId] = useState<string | null>(null);
  const listenersRef = useRef(new Map<SseEventType, Set<(data: unknown) => void>>());
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let disposed = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    // The browser retries transient errors itself but gives up at CLOSED (e.g.
    // daemon restart). Re-auth and open a fresh stream ourselves after a backoff.
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
        } catch { /* ignore */ }
      });

      // Register forwarding for all event types
      const eventTypes: SseEventType[] = [
        'agent_start', 'agent_complete', 'agent_error', 'agent_aborted',
        'step_start', 'thinking', 'tool_call', 'tool_result', 'text_delta',
        'approval_request', 'question', 'context_compaction', 'delegate_event',
        'command_result', 'goal_iteration',
      ];

      for (const eventType of eventTypes) {
        source.addEventListener(eventType, (e) => {
          try {
            const data = JSON.parse((e as MessageEvent).data);
            const handlers = listenersRef.current.get(eventType);
            if (handlers) {
              for (const handler of handlers) {
                handler(data);
              }
            }
          } catch { /* ignore parse errors */ }
        });
      }

      source.onerror = () => {
        setConnected(false);
        // CONNECTING → native retry in flight, leave it. CLOSED → browser gave
        // up, so tear down and re-establish ourselves after a short backoff.
        if (source.readyState === EventSource.CLOSED) {
          source.close();
          if (sourceRef.current === source) sourceRef.current = null;
          // The close may be an expired session; re-bootstrap auth on reconnect.
          resetWebAuth();
          scheduleReconnect();
        }
      };

      source.onopen = () => {
        // connected event will set state
      };
    };

    void connect();

    return () => {
      disposed = true;
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
    () => ({ connected, clientId, addEventListener }),
    [connected, clientId, addEventListener],
  );
}
