import { useEffect, useRef, useState } from 'react';
import { fetchComputer, type ComputerState } from '../api';
import { BrowserSurface } from './BrowserSurface';
import { LOGIN_COPY } from '../shared/ui-copy';
import './ComputerPane.css';

/**
 * The owner signs in to a site in this conversation's own browser, shown live in the chat.
 *
 * What they type goes straight to the page; the agent waits meanwhile and never sees it. With a
 * `keepUrl`, "Keep me signed in" opens the approval page, where one passkey keeps the site for
 * later tasks, and the card closes by itself once it is kept.
 */
export function LoginCard({
  sessionId,
  site,
  keepUrl,
  onDone,
  onDecline,
}: {
  sessionId: string;
  site: string;
  keepUrl: string;
  onDone: () => Promise<void>;
  onDecline: () => Promise<void>;
}) {
  const [view, setView] = useState<ComputerState | null>(null);
  const [error, setError] = useState('');
  const [keeping, setKeeping] = useState(false);
  const [busy, setBusy] = useState(false);
  const inFlight = useRef(false);
  const canKeep = keepUrl.startsWith('https://');

  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    // Pages change on their own while signing in (redirects, codes), so the view keeps refreshing.
    const poll = async () => {
      try {
        const next = await fetchComputer(sessionId);
        if (!disposed) { setView(next); setError(''); }
      } catch (e) {
        if (!disposed) setError(e instanceof Error ? e.message : String(e));
      }
      if (!disposed) timer = setTimeout(poll, 1500);
    };
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, [sessionId]);

  const answer = async (respond: () => Promise<void>) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    try {
      await respond();
    } catch {
      // The hook reports the error. Unlock the card to try again.
      setBusy(false);
    } finally {
      inFlight.current = false;
    }
  };

  return (
    <div className="slide-up" style={{
      margin: '8px 0',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-md)',
      background: 'var(--bg-secondary)',
      overflow: 'hidden',
      position: 'relative',
      zIndex: 20,
    }}>
      <div style={{ padding: '12px 16px 8px' }}>
        <div style={{ fontWeight: 600, fontSize: 14 }}>{LOGIN_COPY.title(site)}</div>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.5, marginTop: 4 }}>
          {keeping ? LOGIN_COPY.keeping : LOGIN_COPY.hint}
        </div>
      </div>
      <div style={{ maxWidth: 430, margin: '0 auto', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
        {view?.state === 'manual' && view.frameId
          ? <BrowserSurface view={view} onView={setView} onTakeControl={() => {}} onError={setError} />
          : <div style={{ padding: 24, fontSize: 13, color: 'var(--text-muted)', textAlign: 'center' }}>{LOGIN_COPY.loading}</div>}
      </div>
      {error && <div role="alert" style={{ padding: '8px 16px', fontSize: 12, color: 'var(--danger)' }}>{error}</div>}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, padding: '10px 16px 14px' }}>
        {canKeep ? (
          <a href={keepUrl} target="_blank" rel="noopener noreferrer" onClick={() => setKeeping(true)} style={{
            flex: 1, padding: '10px 16px', background: 'var(--accent)', color: 'var(--bg-primary)',
            borderRadius: 'var(--radius-md)', fontWeight: 600, fontSize: 14, textAlign: 'center', textDecoration: 'none',
          }}>{LOGIN_COPY.keep}</a>
        ) : (
          <button onClick={() => answer(onDone)} disabled={busy} style={{
            flex: 1, padding: '10px 16px', background: 'var(--accent)', color: 'var(--bg-primary)', border: 'none',
            borderRadius: 'var(--radius-md)', fontWeight: 600, fontSize: 14, opacity: busy ? 0.5 : 1,
          }}>{LOGIN_COPY.done}</button>
        )}
        <button onClick={() => answer(onDecline)} disabled={busy} style={secondary(busy)}>{LOGIN_COPY.notNow}</button>
      </div>
    </div>
  );
}

const secondary = (busy: boolean) => ({
  padding: '10px 16px',
  background: 'transparent',
  color: 'var(--text-secondary)',
  border: '1px solid var(--border)',
  borderRadius: 'var(--radius-md)',
  fontWeight: 600,
  fontSize: 13,
  opacity: busy ? 0.5 : 1,
  cursor: busy ? 'default' : 'pointer',
});
