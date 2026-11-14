import { useEffect, useRef, useState, type ReactNode } from 'react';
import { authFailure, ensureWebAuth, resetWebAuth } from '../api';
import { ConnectionBanner } from './ConnectionBanner';

/** Resolve the account before any conversation or draft can be restored. */
export function SessionBoundary({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [attempt, setAttempt] = useState(0);
  const backoff = useRef(2000);
  const retry = () => {
    resetWebAuth();
    setError(null);
    setAttempt(value => value + 1);
  };

  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    void ensureWebAuth().then(() => {
      if (!disposed) setReady(true);
    }).catch((reason: unknown) => {
      if (disposed) return;
      setError(reason);
      if (!authFailure(reason)) {
        timer = setTimeout(retry, backoff.current);
        backoff.current = Math.min(backoff.current * 2, 15_000);
      }
    });
    return () => { disposed = true; clearTimeout(timer); };
  }, [attempt]);

  if (ready) return children;
  if (!error) return <div role="status" style={{ padding: '16px', color: 'var(--text-secondary)' }}>Connecting to Rune…</div>;
  return <ConnectionBanner authFailure={authFailure(error)} onRetry={retry} />;
}
