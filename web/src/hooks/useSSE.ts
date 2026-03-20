import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SseEventType } from '../types';
import { ensureWebAuth, setClientId as setApiClientId } from '../api';

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

    const connect = async () => {
      try {
        await ensureWebAuth();
      } catch {
        if (!disposed) setConnected(false);
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
        // EventSource auto-reconnects
      };

      source.onopen = () => {
        // connected event will set state
      };
    };

    void connect();

    return () => {
      disposed = true;
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
