import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SSE_EVENT_TYPES, type SseEventType } from '../types';
import { AUTH_EVENT, authFailure, isStaleAuthFailure, eventStreamUrl, type AuthFailure, ensureWebAuth, requireWebAuth, resetWebAuth, setClientId as setApiClientId } from '../api';
import { fetchRunSnapshot, getLiveSessionId } from '../api';
import { RunRecovery } from '../utils/runRecovery';
import { MACHINE_EVENT } from '../utils/presence';

export interface SseConnection {
  connected: boolean;
  authFailure: AuthFailure;
  reconnect: () => void;
  clientId: string | null;
  addEventListener: (event: SseEventType, handler: (data: unknown) => void) => () => void;
  refresh: () => void;
}

export function useSSE(): SseConnection {
  const [authError, setAuthError] = useState<AuthFailure>(null);
  const reconnectRef = useRef<() => void>(() => {});
  const reconnect = useCallback(() => reconnectRef.current(), []);
  const [connected, setConnected] = useState(false);
  const [clientId, setClientId] = useState<string | null>(null);
  const listenersRef = useRef(new Map<SseEventType, Set<(data: unknown) => void>>());
  const sourceRef = useRef<EventSource | null>(null);
  const restoreRef = useRef<() => void>(() => {});
  const refresh = useCallback(() => restoreRef.current(), []);

  useEffect(() => {
    let disposed = false;
    let blocked: AuthFailure = null;
    let snapshotAuthRetried = false;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let recoveryVersion = 0;
    let connectionVersion = 0;
    const emit = (type: SseEventType, data: unknown) => {
      for (const handler of listenersRef.current.get(type) ?? []) handler(data);
    };
    const recovery = new RunRecovery(emit);
    const restore = async () => {
      if (disposed || blocked) return;
      const version = ++recoveryVersion;
      const sessionId = getLiveSessionId();
      recovery.begin();
      try {
        const { run, available } = await fetchRunSnapshot(sessionId);
        if (disposed || version !== recoveryVersion) return;
        snapshotAuthRetried = false;
        if (sessionId !== getLiveSessionId() || !recovery.finish(available === false ? undefined : run)) {
          void restore();
        }
      } catch (error) {
        if (disposed || version !== recoveryVersion) return;
        if (isStaleAuthFailure(error)) { void restore(); return; }
        const failure = authFailure(error);
        if (failure === 'forbidden' || failure && snapshotAuthRetried) {
          stopForAuth(failure);
          return;
        }
        if (failure) { snapshotAuthRetried = true; resetWebAuth(); }
        sourceRef.current?.close();
        sourceRef.current = null;
        setConnected(false);
        scheduleReconnect();
      }
    };

    restoreRef.current = () => { void restore(); };

    // EventSource stops retrying once CLOSED; open a new stream after a backoff.
    const scheduleReconnect = () => {
      if (disposed || blocked || retryTimer) return;
      retryTimer = setTimeout(() => {
        retryTimer = null;
        void connect();
      }, 2000);
    };

    const connect = async () => {
      if (disposed || blocked) return;
      const version = ++connectionVersion;
      try {
        await ensureWebAuth();
      } catch (error) {
        if (!disposed && version === connectionVersion) {
          setConnected(false);
          const failure = authFailure(error);
          if (failure) stopForAuth(failure);
          else scheduleReconnect();
        }
        return;
      }
      if (disposed || version !== connectionVersion) return;

      const source = new EventSource(eventStreamUrl(), { withCredentials: true });
      sourceRef.current = source;
      let announced = false;

      source.addEventListener('connected', (e) => {
        if (disposed || sourceRef.current !== source) return;
        try {
          const data = JSON.parse((e as MessageEvent).data) as { clientId: string };
          announced = true;
          setClientId(data.clientId);
          setApiClientId(data.clientId);
          setAuthError(null);
          setConnected(true);
          void restore();
        } catch { /* ignore */ }
      });

      for (const eventType of SSE_EVENT_TYPES.filter(e => e !== 'connected')) {
        source.addEventListener(eventType, (e) => {
          if (disposed || sourceRef.current !== source) return;
          try {
            const data = JSON.parse((e as MessageEvent).data);
            if (eventType === 'resync_required') void restore();
            else recovery.receive(eventType, data);
          } catch { /* ignore parse errors */ }
        });
      }

      source.onerror = () => {
        if (disposed || sourceRef.current !== source) return;
        setConnected(false);
        // Leave CONNECTING alone while the browser retries.
        if (source.readyState === EventSource.CLOSED) {
          source.close();
          if (sourceRef.current === source) sourceRef.current = null;
          if (!announced) {
            // EventSource hides the HTTP status, so probe a protected endpoint.
            void fetchRunSnapshot(getLiveSessionId()).then(() => {
              if (disposed || version !== connectionVersion) return;
              resetWebAuth();
              scheduleReconnect();
            }).catch((error: unknown) => {
              if (disposed || version !== connectionVersion) return;
              if (isStaleAuthFailure(error)) { scheduleReconnect(); return; }
              const failure = authFailure(error);
              if (failure) stopForAuth(failure);
              else { resetWebAuth(); scheduleReconnect(); }
            });
          } else {
            resetWebAuth();
            scheduleReconnect();
          }
        }
      };

      source.onopen = () => {
        // Wait for `connected` to supply the client ID.
      };
    };

    const stopForAuth = (failure: AuthFailure) => {
      if (!failure || blocked === failure) return;
      blocked = failure;
      recoveryVersion++;
      connectionVersion++;
      if (retryTimer) clearTimeout(retryTimer);
      retryTimer = null;
      sourceRef.current?.close();
      sourceRef.current = null;
      setConnected(false);
      setClientId(null);
      setApiClientId('');
      setAuthError(failure);
      if (failure === 'required') requireWebAuth();
    };
    const onAuth = (event: Event) => stopForAuth((event as CustomEvent<AuthFailure>).detail);
    window.addEventListener(AUTH_EVENT, onAuth);

    const resume = () => {
      if (disposed) return;
      blocked = null;
      snapshotAuthRetried = false;
      setAuthError(null);
      recoveryVersion++;
      sourceRef.current?.close();
      sourceRef.current = null;
      if (retryTimer) clearTimeout(retryTimer);
      retryTimer = null;
      resetWebAuth();
      setConnected(false);
      void connect();
    };
    reconnectRef.current = resume;
    const autoResume = () => { if (!blocked) resume(); };
    const unsubscribeResume = window.rune?.onResume?.(autoResume);
    window.addEventListener('online', autoResume);
    // Reconnect as soon as the hosted machine wakes.
    const onMachine = (event: Event) => {
      if ((event as CustomEvent<string>).detail === 'running' && !sourceRef.current) autoResume();
    };
    window.addEventListener(MACHINE_EVENT, onMachine);

    void connect();

    return () => {
      disposed = true;
      connectionVersion++;
      unsubscribeResume?.();
      window.removeEventListener('online', autoResume);
      window.removeEventListener(AUTH_EVENT, onAuth);
      reconnectRef.current = () => {};
      window.removeEventListener(MACHINE_EVENT, onMachine);
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
    () => ({ connected, authFailure: authError, reconnect, clientId, addEventListener, refresh }),
    [connected, authError, reconnect, clientId, addEventListener, refresh],
  );
}
